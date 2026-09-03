"""Shared constants for nnUNetTrainerENet_27_2_reg_trailing's ("S27.2")
HAWQ per-block W/A search -- the already-trained S27.2 checkpoint
(compression/results.csv's own nnUNetTrainerENet_27_2_reg_trailing row,
stage=27_separable_dense_relu_trailing_array, dice=0.7918, 150 epochs).

S12's own separable_dense_relu recipe (separable_dilated=True on the 4
dilated slots, plain ReLU, use_dsc=False, dsc_no_projection=False) with the
S27 family's own trailing-consolidation addition: context_pattern=
dense_dilation_reg_trailing -- a real, full-rank PROJECTED RegularBottleneck
(reduce->3x3->expand, d=1, symmetric) inserted after each (2,4,8,16)
dilation cycle -- ENet.py's own DENSE_DILATION_REG_TRAILING_PATTERN, {"reg_
bottleneck": True} sentinel. Stage2/3 bottleneck depth bumped from S12's
native 8 to 10 (two full 5-slot (2,4,8,16,reg) cycles) -- confirmed against
results.csv's own ops_flags column for this exact net_name (bottlenecks_
per_stage=4,10,10,2,1).

Native S8.2/S12-family width (channels 4,16,32,16,4). See config_23_1.py
for the full rationale on why this file exists (one source of truth per
architecture, shared across sensitivity.py/finn_stage_costs.py/
block_sensitivity.py/finn_block_costs.py/folding_ilp.py) -- not repeated
here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_27_2_reg_trailing"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 10, 10, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_trailing"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = True
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
