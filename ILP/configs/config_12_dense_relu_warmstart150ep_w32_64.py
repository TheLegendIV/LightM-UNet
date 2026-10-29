"""HALF-width twin of config_12_dense_relu_warmstart150ep (cost estimate only, no checkpoint exists).

Identical recipe -- bottlenecks, bilinear decoder (upsample_conv), dense_dilation, ReLU -- the ONLY change is CHANNELS: (8, 32, 64, 32, 8) = 1/2 of the ENet paper's
(16, 64, 128, 64, 16), versus (4, 16, 32, 16, 4) = 1/4 for the S12 hardware baseline. Layer/site names are identical to the base config; folding and cost are not transferable.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu_warmstart150ep_w32_64"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (8, 32, 64, 32, 8)  # initial, stage1, stage2/3 (context), stage4, stage5
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
