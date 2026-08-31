"""Shared constants for nnUNetTrainerENet_8_1_dscnoproj_context_only's HAWQ
per-block search (compression/results.csv's own 8_reginterleaved_isolation
row, dice=0.8117). Native width (4,16,32,16,4), context_pattern=
dense_dilation_reg_interleaved (bottleneck depth 11, same reg-bookend
structure as config_8_2_relu.py/config_8_2_relu_w24_reg.py), dsc_no_
projection=1 -- but SCOPED to the context stage only
(dsc_no_projection_context_only=1), unlike S8.2's own unscoped dsc_no_
projection=1 (every stage's dilated slots become DSCNoProjectionBottleneck
there). Also, unlike the whole S8.2 family, this net keeps real PReLU
(prelu=1, not the ReLU-only S8.2 recipe) -- USE_PRELU is NOT overridden to
False here, ENet.py's own default (True) is correct and left as-is,
matching every pre-S8.2 config_*.py's own convention.

See config_8_2_relu_w24_no_reg_d2_projected.py for the full USE_PRELU/
DSC_NO_PROJECTION-as-optional-config-globals rationale -- not repeated
here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_1_dscnoproj_context_only"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 11, 11, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_interleaved"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = False
PRELU_VARIANT = "standard"
USE_DSC = False
DSC_NO_PROJECTION = True
DSC_NO_PROJECTION_CONTEXT_ONLY = True
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
