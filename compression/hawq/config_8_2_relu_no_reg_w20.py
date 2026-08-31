"""Shared constants for nnUNetTrainerENet_8_2_relu_no_reg_w20's HAWQ
per-block search (compression/results.csv's own 8_2_relu_no_reg_w20 row,
dice=0.7697). Plain dense_dilation context (no reg-interleaved bookends --
"no reg", native depth 8), dsc_no_projection UNSCOPED ("all" --
regular1/regular4/regular5 also become DSCNoProjectionBottleneck), plain
ReLU, at channels=(4,10,20,10,4) -- named "w20" for its context-stage width
(20). Not divisible by 4 at stage1/stage4 (10) -- buildable directly via
ENet.py's own relaxed channel validation (see that file's updated
comment), no workaround needed here.

See config_8_2_relu_w24_no_reg_d2_projected.py for the full USE_PRELU/
DSC_NO_PROJECTION-as-optional-config-globals rationale -- not repeated
here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_no_reg_w20"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 10, 20, 10, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = False
PRELU_VARIANT = "standard"  # unused: USE_PRELU=False collapses the whole encoder to plain ReLU regardless
USE_PRELU = False
USE_DSC = False
DSC_NO_PROJECTION = True
DSC_NO_PROJECTION_CONTEXT_ONLY = False
REG_BOOKEND_DSC = False

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
