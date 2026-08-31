"""Shared constants for nnUNetTrainerENet_8_2_relu_w24_reg's HAWQ per-block
search (compression/results.csv's own 8_2_relu_w24_projected_slots_ablation
row, dice=0.7764) -- config_8_2_relu.py's own reg-interleaved recipe
(context_pattern=dense_dilation_reg_interleaved, bottleneck depth 11:
[reg,d2,d4,d8,d16] x2 + trailing reg) carried onto config_8_2_relu_w24_no_
reg_d2_projected.py's own widened context (channels 4,12,24,12,4) instead
of the native width (4,16,32,16,4). Every other S8.2 defining trait
unchanged: plain ReLU (use_prelu=False), dsc_no_projection=True (unscoped).

See config_8_2_relu_w24_no_reg_d2_projected.py for the full USE_PRELU/
DSC_NO_PROJECTION-as-optional-config-globals rationale -- not repeated
here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_w24_reg"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 12, 24, 12, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 11, 11, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_interleaved"
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
