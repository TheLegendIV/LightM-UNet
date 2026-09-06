"""Shared constants for nnUNetTrainerENet_27_5_dense_reg_trailing's
("S27.5") HAWQ per-block W/A search -- the already-trained S27.5 checkpoint
(compression/results.csv's own nnUNetTrainerENet_27_5_dense_reg_trailing
row, stage "27_5_dense_reg_trailing", dice=0.7889, 150 epochs).

Byte-for-byte config_27_2_reg_trailing.py's own S27.2 recipe -- same
channels, bottleneck depth (4,10,10,2,1), "dense_dilation_reg_trailing"
context pattern (WITH the trailing consolidation reg-bookend block), no
DSC, plain ReLU -- except the four dilated (2,4,8,16) slots use a plain
dense KxK conv (SEPARABLE_DILATED=False) instead of S27.2's own (K,1)+(1,K)
factored pair. The trailing reg-bookend slot itself is unaffected either
way (always a plain symmetric 3x3, never separable-factored). See
config_23_1.py for the full rationale on why this file exists -- not
repeated here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_27_5_dense_reg_trailing"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 10, 10, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_trailing"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = False
PRELU_VARIANT = "standard"  # unused: USE_PRELU=False collapses the whole encoder to plain ReLU regardless
USE_PRELU = False
USE_DSC = False
DSC_NO_PROJECTION = False
DSC_NO_PROJECTION_CONTEXT_ONLY = False
REG_BOOKEND_DSC = False
DSC_SEPARABLE = False

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
