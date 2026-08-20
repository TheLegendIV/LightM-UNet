"""Shared constants for nnUNetTrainerENet_21_2_u8's HAWQ per-stage W/A
search -- the "U8" (narrowest) width point of the same S19/23_1 recipe
(context_pattern=dense_dilation_reg_interleaved_double_mid,
separable_dilated=1, prelu_variant=nonneg_block, decoder_type=
upsample_conv, use_asymmetric=0, bottlenecks_per_stage=4,12,12,2,1 --
confirmed byte-identical to config_23_1.py except CHANNELS, from
compression/results.csv's nnUNetTrainerENet_21_2_u8 row, stage
21_reginterleaved_separable_nonneg_block_double_mid_width).

See config_23_1.py for the full rationale on why this file exists (one
source of truth per architecture, shared across sensitivity.py/
finn_stage_costs.py) and the 5-stage grouping's own justification -- not
repeated here since it's identical."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_21_2_u8"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 8, 16, 8, 4)  # initial, stage1, stage2/3 (context), stage4, stage5 -- half of 23_1's width
BOTTLENECKS_PER_STAGE = (4, 12, 12, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_interleaved_double_mid"
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
