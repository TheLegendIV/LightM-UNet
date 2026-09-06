"""Shared constants for nnUNetTrainerENet_27_4_reg_trailing_asymmetric's
("S27.4") HAWQ per-block W/A search -- the already-trained S27.4 checkpoint
(compression/results.csv's own nnUNetTrainerENet_27_4_reg_trailing_asymmetric
row, stage=27_4_reg_trailing_asymmetric, dice=0.7863, 150 epochs).

Byte-for-byte config_27_2_reg_trailing.py's own S27.2 recipe -- same
channels, bottleneck depth, decoder, dilation schedule, separable-dilated
dilated slots, no DSC -- except the trailing "consolidation" reg-bookend
slot after each (2,4,8,16) dilation cycle is asymmetric-factored ((3,1)+
(1,3), real ENet-style factoring) instead of a plain symmetric 3x3:
CONTEXT_PATTERN="dense_dilation_reg_trailing_asymmetric" (ENet.py's own
DENSE_DILATION_REG_TRAILING_ASYMMETRIC_PATTERN) with USE_ASYMMETRIC=True
(required for the pattern's "asymmetric" sentinel to actually take effect --
see that pattern's own module comment in ENet.py). See config_23_1.py for
the full rationale on why this file exists (one source of truth per
architecture, shared across sensitivity.py/finn_stage_costs.py/
block_sensitivity.py/finn_block_costs.py/folding_ilp.py) -- not repeated
here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_27_4_reg_trailing_asymmetric"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 10, 10, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_trailing_asymmetric"
USE_ASYMMETRIC = True
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
