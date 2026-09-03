"""MobileNetV3(-Large) trainer -- same env-var-driven pattern as
nnUNetTrainerMobileNetV2.py (see that file's own module docstring for the
full rationale, not repeated here)."""
from __future__ import annotations

import os

from torch import nn

from nnunetv2.nets.MobileNetV3 import MobileNetV3
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
        return MobileNetV3(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            width_mult=float(os.environ.get("MOBILENET_WIDTH_MULT", "1.0")),
        )
