"""ERFNet trainer -- same env-var-driven construction pattern as
nnUNetTrainerENet.py (ENET_CHANNELS etc.), a separate class here because
ERFNet.py's own constructor shape genuinely differs (5-value channels, no
stage2/stage3 split, no use_dsc/use_prelu/decoder_type/... knobs -- see
that file's own module docstring for the full architecture). Reuses
nnUNetTrainerENet's own base class for every training-loop mechanic (LR
schedule, AdamW optimizer, ENET_EPOCHS/ENET_SEED/etc. env vars) UNCHANGED
-- this repo trains every architecture through the SAME single-stage
AdamW/PolyLR pipeline for a controlled, architecture-only comparison, not
ERFNet's own paper training recipe (two-stage encoder-then-decoder,
Adam-with-momentum, LR-halved-on-plateau -- explicitly out of scope, see
ERFNet.py's own module docstring)."""
from __future__ import annotations

import os

from torch import nn

from nnunetv2.nets.ERFNet import ERFNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import _parse_channels, nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerERFNet(nnUNetTrainerENet):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("ERFNet is a 2D architecture. Use the nnU-Net 2d configuration.")
        label_manager = plans_manager.get_label_manager(dataset_json)
        # ERFNet.py's own 5-value convention (initial, stage1, context,
        # stage4, stage5) -- reuses nnUNetTrainerENet.py's own _parse_
        # channels, which already accepts 5 (or 6, unused here) comma-
        # separated ints, same ENET_CHANNELS env var every other trainer
        # in this file reads.
        channels = _parse_channels(os.environ.get("ENET_CHANNELS", "16,64,128,64,16"))
        if len(channels) != 5:
            raise ValueError(
                f"ERFNet needs exactly 5 ENET_CHANNELS values (initial, stage1, context, stage4, stage5) -- "
                f"got {len(channels)}: {channels}."
            )
        network = ERFNet(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            channels=channels,
            stage1_depth=int(os.environ.get("ERFNET_STAGE1_DEPTH", "5")),
            context_depth=int(os.environ.get("ERFNET_CONTEXT_DEPTH", "8")),
            decoder_depth=int(os.environ.get("ERFNET_DECODER_DEPTH", "2")),
            stage1_dropout=float(os.environ.get("ERFNET_STAGE1_DROPOUT", "0.03")),
            context_dropout=float(os.environ.get("ERFNET_CONTEXT_DROPOUT", "0.3")),
            decoder_dropout=float(os.environ.get("ERFNET_DECODER_DROPOUT", "0.0")),
        )
        return network
