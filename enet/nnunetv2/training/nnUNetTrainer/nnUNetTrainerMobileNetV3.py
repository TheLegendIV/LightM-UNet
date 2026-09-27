"""MobileNetV3(-Large/-Small) trainer -- same env-var-driven pattern as
nnUNetTrainerMobileNetV2.py (see that file's own module docstring for the
full rationale, not repeated here).

MOBILENET_VARIANT selects "large" (default, the real base "MobileNetV3-
Large 1.0") or "small" (the real base "MobileNetV3-Small 1.0", per the
paper's own Table 2 -- Howard et al. 2019: 11-block schedule, head2=1024 not
Large's 1280 -- see MobileNetV3.py's own module docstring for the head1-SE
correction history).

MOBILENET_DECODER selects "convtranspose" (default, this repo's own 5-stage
learned-upsample decoder) or "lraspp" (the paper's own Sec 6.4 segmentation
head, Lite R-ASPP, with RF2 tail-channel-halving + dilation-held OS=16 --
see MobileNetV3.py's own LRASPPHead/MobileNetV3 docstrings)."""
from __future__ import annotations

import os

from torch import nn

from nnunetv2.nets.MobileNetV3 import (
    MobileNetV3,
    _LARGE_HEAD1_CHANNELS,
    _LARGE_SETTING,
    _SMALL_HEAD1_CHANNELS,
    _SMALL_HEAD2_CHANNELS,
    _SMALL_SETTING,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerMobileNetV3(nnUNetTrainerENet):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("MobileNetV3 is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        variant = os.environ.get("MOBILENET_VARIANT", "large").lower()
        if variant == "small":
            kwargs = dict(setting=_SMALL_SETTING, head1_channels=_SMALL_HEAD1_CHANNELS,
                          head2_channels=_SMALL_HEAD2_CHANNELS)
        elif variant == "large":
            kwargs = dict(setting=_LARGE_SETTING, head1_channels=_LARGE_HEAD1_CHANNELS,
                          head2_channels=1280)
        else:
            raise ValueError(f"MOBILENET_VARIANT must be 'large' or 'small', got {variant!r}.")
        return MobileNetV3(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            width_mult=float(os.environ.get("MOBILENET_WIDTH_MULT", "1.0")),
            decoder=os.environ.get("MOBILENET_DECODER", "convtranspose"),
            **kwargs,
        )
