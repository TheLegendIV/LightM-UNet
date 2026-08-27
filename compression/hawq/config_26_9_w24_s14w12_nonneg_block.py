"""Shared constants for nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block's
HAWQ per-block W/A search -- new 26_s5_6_probe_family member: 26_5_w24's
own recipe (context stage width bumped 32 -> 24) with stage1/stage4 also
widened 8 -> 12 (CHANNELS=4,8,24,8,4 -> 4,12,24,12,4), and prelu_variant
switched from 26_5_w24's own "standard" (real per-channel PReLU) to
"nonneg_block" (one shared learnable NonNegativePReLU(1) scalar per
bottleneck block) -- deliberate departure so this line is losslessly
FINN-deployable via apply_leaky_slope_overrides, unlike 26_5_w24's own
"standard" recipe (see stage_26_9_w24_s14w12_nonneg_block.job's own header
for the full rationale).

Everything else identical to config_26_5_w24.py: BOTTLENECKS_PER_STAGE=
(4,8,8,2,1), DECODER_TYPE=upsample_conv, CONTEXT_PATTERN=dense_dilation,
USE_ASYMMETRIC=False, SEPARABLE_DILATED=True. See config_23_1.py for the
full rationale on why this file exists (one source of truth per
architecture, shared across sensitivity.py/finn_stage_costs.py) and the
5-stage grouping's own justification -- not repeated here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 12, 24, 12, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = True
PRELU_VARIANT = "nonneg_block"

STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}
STAGE_BOUNDARY_ATTR = {
    "initial": "initial",
    "stage1": "regular1",
    "context": "stage3",
    "stage4": "regular4",
    "stage5": "regular5",
}
STAGE_NAMES = tuple(STAGE_MODULE_ATTRS.keys())
CANDIDATE_BITS = (2, 4, 8)
