import os

import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.SmallENet import SmallENet
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightMUNet import nnUNetTrainerLightMUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

"""
Trainer for nnunetv2.nets.SmallENet.py -- a small, full-resolution, single
logit-channel (sigmoid) ENet variant, purpose-built for the ARCADE grid
patches (dataset-prep/prepare_arcade.py). Only supports binary datasets
(background + exactly one foreground class, e.g. Dataset501_ARCADE_6x6_1c).

Because the network outputs a single logit channel while the dataset's
dataset.json is a plain (non-"regions") binary label dict, nnU-Net's
LabelManager reports has_regions=False and num_segmentation_heads=2 for
this dataset -- the base nnUNetTrainer's loss/validation/final-inference
code all assume that many channels. _build_loss() and validation_step()
below are overridden to force the sigmoid/single-channel path instead (the
same code path nnUNetTrainerLightMUNet already has for has_regions=True
datasets), so per-epoch pseudo-dice logging during training works
correctly. perform_actual_validation() (the final full-resolution
sliding-window export) is NOT compatible with this single-channel output
against a has_regions=False label manager and is left disabled by default
-- see SMALLENET_SKIP_FINAL_VALIDATION below.
"""


class nnUNetTrainerSmallENet(nnUNetTrainerLightMUNet):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = float(os.environ.get("SMALLENET_LR", "1e-3"))
        self.weight_decay = float(os.environ.get("SMALLENET_WEIGHT_DECAY", "1e-2"))
        if os.environ.get("SMALLENET_EPOCHS"):
            self.num_epochs = int(os.environ["SMALLENET_EPOCHS"])
        if os.environ.get("SMALLENET_BATCH_SIZE"):
            self.batch_size = int(os.environ["SMALLENET_BATCH_SIZE"])
        if os.environ.get("SMALLENET_ITERATIONS_PER_EPOCH"):
            self.num_iterations_per_epoch = int(os.environ["SMALLENET_ITERATIONS_PER_EPOCH"])
        if os.environ.get("SMALLENET_VAL_ITERATIONS_PER_EPOCH"):
            self.num_val_iterations_per_epoch = int(os.environ["SMALLENET_VAL_ITERATIONS_PER_EPOCH"])
        if os.environ.get("SMALLENET_DISABLE_CHECKPOINTING", "0") == "1":
            self.disable_checkpointing = True
            self.save_every = 10**9
        elif os.environ.get("SMALLENET_SAVE_EVERY"):
            self.save_every = int(os.environ["SMALLENET_SAVE_EVERY"])
        if os.environ.get("SMALLENET_OUTPUT_FOLDER"):
            self.output_folder = os.environ["SMALLENET_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("SmallENet is a 2D architecture. Use the nnU-Net 2d configuration.")
        if num_input_channels != 1:
            raise ValueError(
                "SmallENet derives its local-contrast-norm channel from a single grayscale "
                f"input channel; this dataset has {num_input_channels} input channels."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if label_manager.has_regions or label_manager.num_segmentation_heads != 2:
            raise ValueError(
                "SmallENet only supports plain binary datasets (background + exactly one "
                f"foreground class). Got has_regions={label_manager.has_regions}, "
                f"num_segmentation_heads={label_manager.num_segmentation_heads}."
            )

        return SmallENet(
            in_channels=1,
            out_channels=1,
            initial_channels=int(os.environ.get("SMALLENET_INITIAL_CHANNELS", "16")),
            stage_channels=int(os.environ.get("SMALLENET_STAGE_CHANNELS", "32")),
            lcn_kernel_size=int(os.environ.get("SMALLENET_LCN_KERNEL", "9")),
        )

    def _build_loss(self):
        return DC_and_BCE_loss(
            {},
            {"batch_dice": self.configuration_manager.batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": self.is_ddp},
            use_ignore_label=self.label_manager.has_ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)
        return optimizer, scheduler

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

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
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()

        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()
            target[target == self.label_manager.ignore_label] = 0
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}

    def perform_actual_validation(self, save_probabilities: bool = False):
        if os.environ.get("SMALLENET_SKIP_FINAL_VALIDATION", "0") == "1":
            self.print_to_log_file("Skipping final full validation because SMALLENET_SKIP_FINAL_VALIDATION=1")
            return
        raise NotImplementedError(
            "SmallENet's single-logit sigmoid output is not wired up to nnU-Net's final "
            "sliding-window validation/export path (it assumes num_segmentation_heads channels "
            "for a has_regions=False label manager). Set SMALLENET_SKIP_FINAL_VALIDATION=1 and "
            "evaluate per-epoch pseudo-dice from training logs, or the labelsPr_* prediction "
            "folders, instead."
        )

    def plot_network_architecture(self):
        if os.environ.get("SMALLENET_SKIP_ARCH_PLOT", "0") == "1":
            self.print_to_log_file("Skipping network architecture plot because SMALLENET_SKIP_ARCH_PLOT=1")
            return
        return super().plot_network_architecture()
