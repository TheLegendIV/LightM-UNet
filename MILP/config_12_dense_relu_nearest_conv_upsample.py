"""Shared constants for nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample's
("S12-dense, nearest+conv decoder") HAWQ per-layer W/A search -- the
already-trained checkpoint (compression/results.csv's own
nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample row, stage
"12_dense_relu_nearest_conv_upsample", dice=0.7779, 143/150 epochs, see
compression/slurm/stage_12_dense_relu_nearest_conv_upsample.job).

Byte-for-byte config_12_dense_relu.py's own recipe (same channels, native
bottleneck depth, plain "dense_dilation" context pattern, no DSC, plain
ReLU, SEPARABLE_DILATED=False) -- the ONLY change is DECODER_TYPE:
"nearest_conv_upsample" instead of "upsample_conv". See ENet.py's
UpsamplingBottleneck for what that changes at the model level (nearest-
neighbor F.interpolate on the decoder's skip/main branch, followed by a
learned 3x3 Conv2d + BatchNorm2d + activation -- the classic "resize-
convolution" anti-checkerboard pattern, applied on top of a bare nearest
resize) -- see config_23_1.py for the full rationale on why this file
exists, not repeated here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "nearest_conv_upsample"
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
CANDIDATE_BITS = (4, 6, 8)
