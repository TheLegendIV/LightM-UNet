"""QAT trainer for nnUNetTrainerENet_8_2_relu_no_reg_d2_projected's own real
HAWQ per-stage bit assignment (compression/hawq/stage_bits_8_2_relu_no_reg_
d2_projected_minres.json: initial/stage1/stage4/stage5 weights=4, context
weights=2, stage1/context/stage4 acts=2, initial/stage5 acts=4 -- the
assignment the real folding search showed feasible at ~30%/15% LUT/BRAM,
185ms latency, see this session's own transcript).

Deliberately narrow, same philosophy every per-checkpoint quant trainer in
this repo already uses: every architecture axis (CHANNELS, BOTTLENECKS_
PER_STAGE, CONTEXT_PATTERN, USE_PRELU, DSC_NO_PROJECTION) is hardcoded to
this specific FP32 checkpoint's own architecture (see compression/hawq/
config_8_2_relu_no_reg_d2_projected.py), not read from generic ENET_*
env vars -- there is exactly one real trained FP32 checkpoint this could
possibly warm-start from.

Built on CombinedQuantENet (enet/nnunetv2/nets/CombinedQuantENet.py) --
this architecture's own QuantDSCNoProjectionBottleneck/QuantRegularBottleneck
mix, extended this session to support context_pattern="dense_dilation_
d2_projected" (see that file's own module docstring/DENSE_DILATION_D2_
PROJECTED_PATTERN import). Per-BLOCK bit dicts CombinedQuantENet's own
constructor needs are derived from the per-STAGE assignment via
expand_stage_bits (broadcasts one (w,a) pair per HAWQ stage group to every
block within it) -- this architecture has no per-block bit granularity of
its own, only 5 HAWQ stage groups.

No leaky-slope machinery at all (unlike the 26_9_w24_s14w12_nonneg_block
family's own quant trainers) -- this architecture is plain ReLU throughout
(negative_slope=None everywhere), so there is no alpha/quant_enabled/
calibration-of-the-leaky-chain complexity to carry here; only the
QuantDSCNoProjectionBottleneck/QuantRegularBottleneck activation/weight
quantizers (Brevitas PARAMETER_FROM_STATS scaling) need calibration, done
BEFORE this trainer ever runs, by compression/post-quantization/
calibrate_8_2_relu_no_reg_d2_projected.py (confirmed load-bearing by this
session's own local proxy test: loss trend was noisy/non-monotonic without
calibration, clean and monotonically decreasing with it) -- this trainer
always warm-starts from that ALREADY-CALIBRATED checkpoint via
ENET_PRETRAINED_CHECKPOINT, never a raw FP32 one.
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
CONTEXT_PATTERN = "dense_dilation_d2_projected"
STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}


class nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_d2_projected(nnUNetTrainerENet):
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
                "nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_d2_projected is a 2D architecture. "
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
