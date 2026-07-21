"""Trainer for ENetPostRefinement (nnunetv2.nets.ENetPostRefinement.py) --
the "separate stems, fuse later" sibling of nnUNetTrainerENetPost. Same
dataset (Dataset509_ARCADE_ENetPost), same clDice-augmented loss (inherited
unchanged from nnUNetTrainerENetPost -- see that file's docstring for the
loss rationale), only build_network_architecture differs: builds
ENetPostRefinement (two ENet InitialBlock stems, one per channel, fused via
a 1x1 conv) instead of plain ENet with in_channels=2 (which mixes both
channels in a single shared first conv).

See ENetPostRefinement.py's module docstring for why this variant exists --
short version: Dataset509's channel 1 is a prior guess from another model
(nnUNetTrainerSmallENet trained on Dataset507), not an independent
modality, and early fusion gives the network unrestricted immediate access
to lean on that guess from the first conv. Separate stems don't prevent
that, but they guarantee the raw-image pathway develops real features
before the probability channel can dominate.

ENETPOST_STEM_CHANNELS (default: max(2, initial_channels // 2), i.e. half
of ENET_CHANNELS' first value) controls each stem's own channel width
before fusion -- see ENetPostRefinement.__init__.
"""
import os

from torch import nn

from nnunetv2.nets.ENetPostRefinement import ENetPostRefinement
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import _parse_channels
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENetPost import nnUNetTrainerENetPost
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerENetPostRefinement(nnUNetTrainerENetPost):

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("ENetPostRefinement is a 2D architecture. Use the nnU-Net 2d configuration.")
        if num_input_channels != 2:
            raise ValueError(
                "ENetPostRefinement takes exactly 2 input channels (raw image, upstream-model "
                f"probability); this dataset has {num_input_channels}. Use Dataset509_ARCADE_ENetPost "
                "(dataset-prep/prepare_arcade_509_enetpost.py) or another dataset with the same "
                "channel_names convention."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = _parse_channels(os.environ.get("ENET_CHANNELS", "20,72,144,72,20"))
        stem_channels_env = os.environ.get("ENETPOST_STEM_CHANNELS")
        stem_channels = int(stem_channels_env) if stem_channels_env else None

        return ENetPostRefinement(
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            stem_channels=stem_channels,
        )
