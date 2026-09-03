"""MobileNetV2 trainer -- same env-var-driven pattern as
nnUNetTrainerERFNet.py (see that file's own module docstring for the full
rationale, not repeated here): a separate class because MobileNetV2.py's
own constructor shape (width_mult, not this repo's usual 5-value channels)
genuinely differs, reusing nnUNetTrainerENet's own training-loop mechanics
(LR schedule, AdamW optimizer, ENET_EPOCHS/ENET_SEED/etc. env vars)
UNCHANGED -- same single-stage AdamW/PolyLR pipeline every other
architecture in this repo trains through, not MobileNetV2's own original
ImageNet classification recipe (RMSProp, exponential LR decay, dropout,
label smoothing -- out of scope here, same "architecture-only comparison"
rationale as every other model file added this session)."""
from __future__ import annotations

import os

from torch import nn

from nnunetv2.nets.MobileNetV2 import MobileNetV2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerMobileNetV2(nnUNetTrainerENet):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("MobileNetV2 is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        return MobileNetV2(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            width_mult=float(os.environ.get("MOBILENET_WIDTH_MULT", "1.0")),
        )
