import os
import random

import numpy as np
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.nets.TVResNet101UNet import TVResNet101UNet
from torch.optim import AdamW


class nnUNetTrainerTVResNet101UNet(nnUNetTrainerNoDeepSupervision):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.grad_scaler = None
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        self.encoder_weights = "IMAGENET1K_V2"

        # Shared env-var hooks (same pattern as nnUNetTrainerENet.py /
        # nnUNetTrainerLightMUNet.py) -- this trainer extends
        # nnUNetTrainerNoDeepSupervision directly (not the LightMUNet base),
        # so it previously had zero epoch-count/seed/etc. control of its own.
        if os.environ.get("RESNET101_SEED"):
            # nnU-Net's base trainer only seeds the train/val SPLIT
            # (np.random.RandomState(12345+fold), see do_split) -- weight
            # init, augmentation, and dataloader-worker randomness are never
            # explicitly seeded anywhere upstream. Set as early as possible
            # (before .initialize() builds the network) so it's
            # reproducible, not just "whatever the ambient RNG state was".
            resnet101_seed = int(os.environ["RESNET101_SEED"])
            random.seed(resnet101_seed)
            np.random.seed(resnet101_seed)
            torch.manual_seed(resnet101_seed)
            torch.cuda.manual_seed_all(resnet101_seed)
            self.resnet101_seed = resnet101_seed
        else:
            self.resnet101_seed = None
        if os.environ.get("RESNET101_EPOCHS"):
            self.num_epochs = int(os.environ["RESNET101_EPOCHS"])
        if os.environ.get("RESNET101_BATCH_SIZE"):
            self.batch_size = int(os.environ["RESNET101_BATCH_SIZE"])
        if os.environ.get("RESNET101_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["RESNET101_ITERATIONS_PER_EPOCH"])
        if os.environ.get("RESNET101_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["RESNET101_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("RESNET101_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        if os.environ.get("RESNET101_OUTPUT_FOLDER"):
            self.output_folder = os.environ["RESNET101_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)
            # See nnUNetTrainerENet.py's identical fix: nnUNetTrainer.__init__
            # (called above via super().__init__()) already derived
            # self.log_file from self.output_folder's DEFAULT value before
            # this override runs, and never re-derives it later -- relocate
            # the same timestamped filename the base class already picked
            # into the right folder so the human-readable log lands next to
            # this run's own checkpoints instead of a bare shared default.
            self.log_file = os.path.join(self.output_folder, os.path.basename(self.log_file))

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        if len(configuration_manager.patch_size) != 2:
            raise ValueError("nnUNetTrainerTVResNet101UNet only supports 2D configurations.")

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = TVResNet101UNet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            encoder_weights="IMAGENET1K_V2",
        )

        return model

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerTVResNet101UNetNoPretrain(nnUNetTrainerTVResNet101UNet):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        if len(configuration_manager.patch_size) != 2:
            raise ValueError("nnUNetTrainerTVResNet101UNetNoPretrain only supports 2D configurations.")

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = TVResNet101UNet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            encoder_weights=None,
        )

        return model
