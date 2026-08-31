"""QAT trainer for nnUNetTrainerENet_8_2_relu_no_reg_fullwidth's own real
HAWQ per-stage bit assignment (compression/hawq/stage_bits_8_2_relu_no_reg_
fullwidth_minres.json: initial/stage1/stage4/stage5 weights=4, context
weights=2, stage1/context/stage4 acts=2, initial acts=8, stage5 acts=4 --
the assignment the real folding search showed feasible at ~31%/15% LUT/
BRAM, 152.9ms latency at the 100pct cap, see this session's own
transcript).

Same construction as nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_
d2_projected.py (see that file's own module docstring for the full
rationale -- not repeated here) -- this trainer differs only in
CONTEXT_PATTERN ("dense_dilation", plain, no d2-projection, matching this
architecture's own FP32 checkpoint) and always warm-starts from
compression/post-quantization/calibrate_8_2_relu_no_reg_fullwidth.py's own
calibrated checkpoint via ENET_PRETRAINED_CHECKPOINT, never the raw FP32
one directly (calibration is load-bearing here too, same as its sibling).
"""
from __future__ import annotations

import json
import os

from torch import nn

from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet, block_names_for, expand_stage_bits
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerENet import nnUNetTrainerENet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}


class nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth(nnUNetTrainerENet):
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Same Brevitas parameter-scaling re-materialization quirk
        nnUNetTrainerENetQuant.py/nnUNetTrainerENetQuant26_5_w24Block.py/
        nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock.py/
        nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_d2_projected.py all
        already document and fix -- only bites a resumed (--c) run
        (confirmed by direct repro on that sibling trainer this session:
        resuming via --c threw a cuda/cpu device-mismatch error inside
        Brevitas's own quant conv forward without this fix)."""
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
                "nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth is a 2D architecture. "
                "Use the nnU-Net 2d configuration."
            )
        label_manager = plans_manager.get_label_manager(dataset_json)
        if num_input_channels != 1 or label_manager.num_segmentation_heads != 5:
            raise ValueError(
                f"This trainer is hardcoded to in_channels=1, out_channels=5 (Dataset509_ARCADE_1x1_4c) -- "
                f"got num_input_channels={num_input_channels}, num_segmentation_heads="
                f"{label_manager.num_segmentation_heads}. Wrong dataset/plans for this trainer."
            )

        stage_bits_file = os.environ.get("ENET_STAGE_BITS_FILE")
        if not stage_bits_file:
            raise ValueError(
                "ENET_STAGE_BITS_FILE must point to a stage_bits_*.json (compression/hawq/ilp_search.py's own "
                "output format: {'stage_weight_bits': {...}, 'stage_act_bits': {...}}, one entry per "
                f"{list(STAGE_MODULE_ATTRS)})."
            )
        with open(stage_bits_file) as f:
            stage_bits = json.load(f)
        block_names = block_names_for(BOTTLENECKS_PER_STAGE)
        block_weight_bits, block_act_bits = expand_stage_bits(
            stage_bits["stage_weight_bits"], stage_bits["stage_act_bits"], STAGE_MODULE_ATTRS, block_names,
        )

        pretrained_checkpoint = os.environ.get("ENET_PRETRAINED_CHECKPOINT")
        common_kwargs = dict(
            out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
            context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
            use_dsc=False, dsc_no_projection=True, separable_dilated=False, trainable_slope=False,
        )
        if pretrained_checkpoint:
            model = CombinedQuantENet.from_pretrained(
                pretrained_checkpoint, block_weight_bits, block_act_bits, **common_kwargs,
            )
        else:
            model = CombinedQuantENet(block_weight_bits, block_act_bits, **common_kwargs)
        return model
