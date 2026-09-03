"""QAT trainer for nnUNetTrainerENet_27_2_reg_trailing's own real per-BLOCK
HAWQ bit assignment (compression/hawq/artifacts/block_bits_27_2_reg_
trailing_min4.json -- one independent (weight_bits, act_bits) pair per
individual bottleneck block, restricted to CANDIDATE_BITS={4,8} only, no
2-bit option anywhere -- ilp_search.py's own --candidate-bits 4,8 override,
same "min4" convention every other min4 job in this repo uses. Real
folding at this bit-width is Optimal at a 100% hard LUT/BRAM cap: 100.0%
LUT / 28.0% BRAM / ~174.5ms @ 100MHz -- compression/hawq/artifacts/folding_
block_27_2_reg_trailing_min4_hardcap100_maxspeed.json).

Same construction/Brevitas-device-fix pattern as every other
CombinedQuantENet_*_perblock trainer (see e.g.
nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock's own
module docstring for the full rationale -- not repeated here) -- this
trainer differs from S12's own perblock trainer only in architecture:
context_pattern="dense_dilation_reg_trailing" (a real projected
RegularBottleneck trailing-consolidation block after each dilation cycle,
CombinedQuantENet.py's first use of this pattern -- required adding it to
that file's own VALID_CONTEXT_PATTERNS/dispatch this session) and
bottlenecks_per_stage=(4,10,10,2,1) (stage2/3 depth 10, not S12's native
8). Always warm-starts from compression/post-quantization/calibrate_27_2_
reg_trailing_perblock.py's own calibrated checkpoint via
ENET_PRETRAINED_CHECKPOINT.
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 10, 10, 2, 1)
CONTEXT_PATTERN = "dense_dilation_reg_trailing"


class nnUNetTrainerCombinedQuantENet_27_2_reg_trailing_perblock(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk every
        other CombinedQuantENet trainer this session already documents and
        fixes -- only bites a resumed (--c) run. See
        nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth_perblock.py's
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
                "nnUNetTrainerCombinedQuantENet_27_2_reg_trailing_perblock is a 2D architecture. "
                "Use the nnU-Net 2d configuration."
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
            use_dsc=False, dsc_no_projection=False, separable_dilated=True, trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = CombinedQuantENet.from_pretrained(
                pretrained_checkpoint, block_weight_bits, block_act_bits, **common_kwargs,
            )
        else:
            model = CombinedQuantENet(block_weight_bits, block_act_bits, **common_kwargs)
        return model
