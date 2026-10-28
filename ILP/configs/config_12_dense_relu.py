"""Shared constants for nnUNetTrainerENet_12_dense_relu's ("S12-dense")
HAWQ per-block W/A search -- the already-trained S12-dense checkpoint
(compression/results.csv's own nnUNetTrainerENet_12_dense_relu row, stage
"12_dense_relu", dice=0.7876, 147/150 epochs).

Byte-for-byte config_12_separable_dense_relu.py's own S12 recipe -- same
channels, native bottleneck depth (4,8,8,2,1), plain "dense_dilation"
context pattern (no reg-trailing block), no DSC, plain ReLU -- except the
four dilated (2,4,8,16) slots use a plain dense KxK conv (SEPARABLE_DILATED=
False) instead of S12's own (K,1)+(1,K) factored pair. See config_23_1.py
for the full rationale on why this file exists (one source of truth per
architecture, shared across sensitivity.py/finn_stage_costs.py/
block_sensitivity.py/finn_block_costs.py/folding_ilp.py) -- not repeated
here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
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
