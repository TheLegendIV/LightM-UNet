"""Shared constants for nnUNetTrainerENet_8_2_relu_no_reg_d2_projected's HAWQ
per-stage search (compression/results.csv's own
8_2_relu_no_reg_d2_projected_fullwidth row, dice=0.794). S8.2's own defining
traits (ReLU, dsc_no_projection=1 unscoped) at S8.2's own NATIVE width
(channels 4,16,32,16,4 -- same width as config_8_2_relu.py, NOT the width-24
config_8_2_relu_w24_no_reg_d2_projected.py), but with context_pattern=
dense_dilation_d2_projected (plain dense_dilation's 8-slot layout, d=2
becomes a real dilated projected RegularBottleneck restoring cross-channel
mixing, d=4/d=8/d=16 stay DSCNoProjectionBottleneck) instead of S8.2's own
dense_dilation_reg_interleaved (bottleneck depth 11).

See config_8_2_relu_w24_no_reg_d2_projected.py for the full USE_PRELU/
DSC_NO_PROJECTION-as-optional-config-globals rationale -- not repeated
here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_no_reg_d2_projected"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_d2_projected"
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
