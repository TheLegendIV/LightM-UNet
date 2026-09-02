"""QAT trainer for nnUNetTrainerENet_8_2_relu_w16_no_reg_context_only's own
real per-BLOCK HAWQ bit assignment -- generic, bit-assignment-agnostic
(reads ENET_BLOCK_BITS_FILE at runtime), same pattern as
nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_w16_perblock (see that
file's own module docstring for the full rationale -- not repeated here).

Differs from that UNSCOPED sibling in exactly one architecture flag:
dsc_no_projection_context_only=True (CombinedQuantENet's own scoping knob,
see nets/CombinedQuantENet.py) -- only stage2/stage3's dilated slots become
DSCNoProjectionBottleneck here; regular1/regular4/regular5 stay REAL
projected RegularBottleneck blocks (matching the FP32 source net's own
"rest is normal ENet style" architecture). Same channels=(4,8,16,8,4)
("w16" width) otherwise.
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 8, 16, 8, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"


class nnUNetTrainerCombinedQuantENet_8_2_relu_w16_no_reg_context_only_perblock(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other CombinedQuantENet trainer this session already documents and
        fixes -- only bites a resumed (--c) run. See
        nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_w20_perblock.py's
        own load_checkpoint for the full repro/rationale."""
        super().load_checkpoint(filename_or_checkpoint)
        self.network = self.network.to(self.device)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError(
                "nnUNetTrainerCombinedQuantENet_8_2_relu_w16_no_reg_context_only_perblock is a 2D "
                "architecture. Use the nnU-Net 2d configuration."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"This trainer is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads="
                f"{label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        block_bits_file = os.environ.get("ENET_BLOCK_BITS_FILE")
        if not block_bits_file:
            raise ValueError(
                "ENET_BLOCK_BITS_FILE must point to a block_bits_*.json (compression/hawq/ilp_search.py's own "
                "output format run at block granularity: {'stage_weight_bits': {...}, 'stage_act_bits': {...}}, "
                "one entry per individual bottleneck block name -- see block_utils.enumerate_blocks)."
            )
        with open(block_bits_file) as f:
            block_bits = json.load(f)
        block_weight_bits = block_bits["stage_weight_bits"]
        block_act_bits = block_bits["stage_act_bits"]

        pretrained_checkpoint = os.environ.get("ENET_PRETRAINED_CHECKPOINT")
        common_kwargs = dict(
            out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
            context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
            use_dsc=False, dsc_no_projection=True, dsc_no_projection_context_only=True,
            separable_dilated=False, trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = CombinedQuantENet.from_pretrained(
                pretrained_checkpoint, block_weight_bits, block_act_bits, **common_kwargs,
            )
        else:
            model = CombinedQuantENet(block_weight_bits, block_act_bits, **common_kwargs)
        return model
