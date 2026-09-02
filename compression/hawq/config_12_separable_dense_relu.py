"""Shared constants for nnUNetTrainerENet_12_separable_dense_relu's ("S12.1")
HAWQ per-block W/A search -- the already-trained S12.1 checkpoint
(compression/results.csv's own nnUNetTrainerENet_12_separable_dense_relu
row, stage=12_separable_dense_relu, dice=0.7729, 144 epochs). S5.6's own
separable_dilated+dense_dilation recipe (see config_5_6.py) but with plain
ReLU (use_prelu=0) instead of per-channel PReLU -- i.e. separable_dilated
applied to a RELU network, not DSC at all (use_dsc=0, dsc_no_projection=0
throughout -- confirmed against results.csv's own ops_flags column for this
exact net_name).

Native S8.2/S5.6-family width (channels 4,16,32,16,4), same depth
(4,8,8,2,1) as S5.6. See config_23_1.py for the full rationale on why this
file exists (one source of truth per architecture, shared across
sensitivity.py/finn_stage_costs.py/block_sensitivity.py/finn_block_costs.py/
folding_ilp.py) -- not repeated here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_separable_dense_relu"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
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
