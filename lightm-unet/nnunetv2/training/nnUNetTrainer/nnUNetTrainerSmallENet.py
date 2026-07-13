import os
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
from torch import nn
from torch.optim import AdamW

from nnunetv2.nets.SmallENet import SmallENet
from nnunetv2.training.data_augmentation.custom_transforms.vessel_gap_transform import VesselGapTransform
from nnunetv2.training.loss.cldice import DC_and_BCE_and_clDice_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
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

SmallENet always takes exactly 1 input channel (in_channels=1 is hardcoded
in build_network_architecture's SmallENet(...) call below -- it's not
derived from the dataset). SMALLENET_INPUT_CHANNELS (default "0") selects
which channel index/indices of a multi-channel dataset to actually feed the
network -- e.g. Dataset507_ARCADE_refinement declares 2 channels
(grayscale, predicted_mask) for the *planned* two-channel refinement net,
but this trainer only supports 1, so by default it just uses channel 0
(grayscale) and ignores predicted_mask. Sliced in train_step/
validation_step, right after data moves to device; build_network_architecture
only checks the dataset actually *has* the requested channel(s), it doesn't
change how many the network is built with.

get_training_transforms() is overridden to optionally splice in
VesselGapTransform (see
nnunetv2/training/data_augmentation/custom_transforms/vessel_gap_transform.py
and dataset-prep/preview_augmentations.ipynb) -- off by default, enable with
SMALLENET_GAP_AUG=1.

_build_loss() uses DC_and_BCE_and_clDice_loss (nnunetv2/training/loss/cldice.py)
-- Dice+BCE plus a soft-clDice term rewarding topological connectivity, same
idea as nnUNetTrainerENetComboClDice but adapted for this trainer's single-
logit sigmoid output (do_bg=True: there's no separate background channel to
exclude, the one output channel *is* the foreground). weight_cldice defaults
to 1.0 -- the same weight as Dice and BCE, not a token amount -- so it
actually competes for gradient, not diluted noise. Tune with
SMALLENET_CLDICE_WEIGHT (0.0 reproduces the plain Dice+BCE loss exactly) and
SMALLENET_CLDICE_ITERS (default 12; see cldice.py's soft_skeletonize
docstring for why -- num_iter needs to cover roughly half the widest vessel's
width in pixels or that region contributes nothing to the loss).
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
        self.cldice_weight = float(os.environ.get("SMALLENET_CLDICE_WEIGHT", "1.0"))
        self.cldice_num_iter = int(os.environ.get("SMALLENET_CLDICE_ITERS", "12"))
        self.input_channels = [int(x) for x in os.environ.get("SMALLENET_INPUT_CHANNELS", "0").split(",")]
        if os.environ.get("SMALLENET_OUTPUT_FOLDER"):
            self.output_folder = os.environ["SMALLENET_OUTPUT_FOLDER"]
            self.output_folder_base = os.path.dirname(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)
            # super().__init__() already computed self.log_file from the default
            # (pre-override) output_folder, so it must be redirected here too --
            # otherwise the training log and the checkpoints silently end up in
            # two different directories whenever SMALLENET_OUTPUT_FOLDER differs
            # from nnU-Net's default path (e.g. a stale/differently-cased
            # $nnUNet_results in the shell that exported it).
            timestamp = datetime.now()
            self.log_file = os.path.join(
                self.output_folder,
                "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt"
                % (
                    timestamp.year, timestamp.month, timestamp.day,
                    timestamp.hour, timestamp.minute, timestamp.second,
                ),
            )

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
        selected_channels = [int(x) for x in os.environ.get("SMALLENET_INPUT_CHANNELS", "0").split(",")]
        if len(selected_channels) != 1:
            raise ValueError(
                f"SmallENet takes exactly 1 input channel; SMALLENET_INPUT_CHANNELS={selected_channels} "
                "selects more/fewer than one."
            )
        if max(selected_channels) >= num_input_channels:
            raise ValueError(
                f"SMALLENET_INPUT_CHANNELS={selected_channels} selects a channel not present in this "
                f"dataset (it only has {num_input_channels})."
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

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data, order_resampling_seg, border_val_seg, use_mask_for_norm,
            is_cascaded, foreground_labels, regions, ignore_label,
        )
        gap_transform = VesselGapTransform.from_env()
        if gap_transform is not None:
            # Insert right after the geometric/appearance augmentations (spatial warp,
            # mirror, ...) so data and seg are already co-registered in their final
            # orientation, and before RemoveLabelTransform/RenameTransform so 'seg'
            # still holds raw class-index labels under its original key.
            insert_at = next(
                (i for i, t in enumerate(tr_transforms.transforms) if isinstance(t, RemoveLabelTransform)),
                len(tr_transforms.transforms),
            )
            tr_transforms.transforms.insert(insert_at, gap_transform)
        return tr_transforms

    def _build_loss(self):
        return DC_and_BCE_and_clDice_loss(
            bce_kwargs={},
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": self.is_ddp,
            },
            cldice_kwargs={"num_iter": self.cldice_num_iter, "do_bg": True},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_cldice=self.cldice_weight,
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

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)[:, self.input_channels]
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

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)[:, self.input_channels]
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
