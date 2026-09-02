"""Shared constants for nnUNetTrainerENet_8_2_relu_no_reg_separable_dsc's
HAWQ per-stage search -- IDENTICAL to config_8_2_relu_no_reg_fullwidth.py
(S8.2's own defining traits: ReLU, dsc_no_projection=1 unscoped, native
width channels 4,16,32,16,4, context_pattern=dense_dilation) EXCEPT one new
flag: dsc_separable=True (ENet.py Stage 18) additionally factors
DSCNoProjectionBottleneck's own depthwise KxK pass into a (K,1)+(1,K)
depthwise PAIR -- a second, independent factoring axis on top of DSC's
existing channel-wise (depthwise/pointwise) one, applied at EVERY dilation
rate (unlike separable_dilated, which is a no-op at dilation=1 for
RegularBottleneck's own conv -- see DSCNoProjectionBottleneck's own
docstring for why no such alternative exists here). No checkpoint exists
yet for this architecture (see compression/hawq/hypothetical_8_2_relu_no_
reg_separable_dsc_int4.py for a pre-training uniform-INT4 folding-ILP cost
estimate) -- this config file is scaffolding for the REAL per-block HAWQ
search (block_sensitivity.py/finn_block_costs.py/ilp_search.py) once a real
FP32 checkpoint exists, matching every other config_*.py's own purpose.

See config_8_2_relu_no_reg_d2_projected.py for the full USE_PRELU/
DSC_NO_PROJECTION-as-optional-config-globals rationale -- not repeated
here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_no_reg_separable_dsc"
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
DSC_NO_PROJECTION = True
DSC_NO_PROJECTION_CONTEXT_ONLY = False
REG_BOOKEND_DSC = False
DSC_SEPARABLE = True  # the one difference from config_8_2_relu_no_reg_fullwidth.py

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
