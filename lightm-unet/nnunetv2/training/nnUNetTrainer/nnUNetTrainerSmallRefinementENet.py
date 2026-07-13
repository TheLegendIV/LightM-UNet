import os

import torch
from torch import nn

from nnunetv2.nets.SmallRefinementENet import SmallRefinementENet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSmallENet import nnUNetTrainerSmallENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

"""
Trainer for nnunetv2.nets.SmallRefinementENet.py -- the two-channel
(raw image + first-pass predicted mask) refinement network planned in
analysis/507_refinement_net_plan.md, for Dataset507_ARCADE_refinement.

Everything (_build_loss, get_training_transforms/VesselGapTransform,
checkpoint-path fix, clDice weight, etc.) is inherited unchanged from
nnUNetTrainerSmallENet -- the only differences are build_network_architecture
(builds SmallRefinementENet instead of SmallENet, and does NOT restrict the
dataset to 1 input channel) and forcing self.input_channels to [0, 1] so the
inherited train_step/validation_step's `data[:, self.input_channels]` slice
is a no-op that keeps both channels, instead of nnUNetTrainerSmallENet's
default of channel 0 only (see SMALLENET_INPUT_CHANNELS in that trainer's
docstring -- this subclass always wants both, so it isn't left configurable
here the way it is for the single-channel baseline trainer).
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
