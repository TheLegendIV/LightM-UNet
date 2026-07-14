import os

import torch
from torch import nn

from nnunetv2.nets.SmallRefinementENet import SmallRefinementENet
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSmallENet import nnUNetTrainerSmallENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

"""
Trainer for nnunetv2.nets.SmallRefinementENet.py -- the two-channel
(raw image + first-pass predicted mask) refinement network planned in
analysis/507_refinement_net_plan.md, for Dataset507/508_ARCADE_refinement*.

Most of this (_build_loss, get_training_transforms/VesselGapTransform,
checkpoint-path fix, clDice weight, etc.) is inherited unchanged from
nnUNetTrainerSmallENet -- build_network_architecture (builds
SmallRefinementENet instead of SmallENet, and does NOT restrict the dataset
to 1 input channel) and forcing self.input_channels to [0, 1] so the
inherited train_step/validation_step's `data[:, self.input_channels]` slice
is a no-op that keeps both channels, instead of nnUNetTrainerSmallENet's
default of channel 0 only (see SMALLENET_INPUT_CHANNELS in that trainer's
docstring -- this subclass always wants both, so it isn't left configurable
here the way it is for the single-channel baseline trainer).

Gap-focused BCE weighting (train_step/validation_step ARE overridden for
this): a refinement net can reach a deceptively high pooled dice by mostly
copying channel 1 (the input predicted mask) through -- correct on most of
the patch by construction, since even a "broken" patch's break is small
relative to the whole 96x96 area. Standard Dice+BCE+clDice, averaged over
every pixel, only weakly penalizes getting the (spatially tiny) actually-
wrong pixels wrong. To counter that: every train/val step computes a
disagreement mask between channel 1 (thresholded) and GT, and upweights the
BCE loss by SMALLENET_GAP_LOSS_BOOST (default 8.0) specifically on
disagreeing pixels -- i.e. the network is judged far more harshly on exactly
the pixels a passthrough gets wrong, not on the ones it gets right for free.
SMALLENET_GAP_LOSS_BOOST=0 reproduces the unweighted loss exactly.

Threshold caveat: channel 1 reaches this code already z-score normalized
(baked into the preprocessed .npy at preprocessing time, not raw
probability) -- there's no access to the original per-case mean/std here to
recover an exact "probability > 0.5" cut. Thresholding the normalized
channel at 0 (i.e. "above this case's own mean") is used as a monotonic
proxy instead: z-score normalization preserves ordering, and the
probability channel is strongly bimodal (background near 0, confident
vessel near 1) for essentially every patch, so 0 falls in the gap between
the two modes for the cases that matter. Not exactly calibrated to 0.5 in
raw-probability terms, but doesn't need to be -- it only has to separate
"this pixel looked like predicted-vessel" from "predicted-background",
which a bimodal distribution's own mean does reliably.
"""


class nnUNetTrainerSmallRefinementENet(nnUNetTrainerSmallENet):

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
        self.input_channels = [0, 1]
        self.gap_loss_boost = float(os.environ.get("SMALLENET_GAP_LOSS_BOOST", "8.0"))

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("SmallRefinementENet is a 2D architecture. Use the nnU-Net 2d configuration.")
        if num_input_channels != 2:
            raise ValueError(
                "SmallRefinementENet takes exactly 2 input channels (raw image, predicted_mask); "
                f"this dataset has {num_input_channels}. Use Dataset507_ARCADE_refinement "
                "(dataset-prep/prepare_arcade_507_refinement.py) or another dataset with the same "
                "channel_names convention."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if label_manager.has_regions or label_manager.num_segmentation_heads != 2:
            raise ValueError(
                "SmallRefinementENet only supports plain binary datasets (background + exactly one "
                f"foreground class). Got has_regions={label_manager.has_regions}, "
                f"num_segmentation_heads={label_manager.num_segmentation_heads}."
            )

        return SmallRefinementENet(
            stem_channels=int(os.environ.get("SMALLENET_STEM_CHANNELS", "8")),
            stage_channels=int(os.environ.get("SMALLENET_STAGE_CHANNELS", "32")),
        )

    def _gap_pixel_weight(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        """1 + gap_loss_boost wherever channel 1 (predicted mask, thresholded)
        disagrees with GT -- see module docstring for the normalization
        caveat. None (no weighting) if SMALLENET_GAP_LOSS_BOOST=0."""
        if self.gap_loss_boost == 0:
            return None
        pred_mask_proxy = data[:, 1:2] > 0
        gt_binary = target > 0.5
        disagreement = (pred_mask_proxy != gt_binary).float()
        return 1.0 + self.gap_loss_boost * disagreement

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
        pixel_weight = self._gap_pixel_weight(data, target)
        l = self.loss(output, target, pixel_weight=pixel_weight)
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
        pixel_weight = self._gap_pixel_weight(data, target)
        l = self.loss(output, target, pixel_weight=pixel_weight)

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
