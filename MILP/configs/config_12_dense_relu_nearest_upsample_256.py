# 256x256 variant of config_12_dense_relu_nearest_upsample (U4 widths (4, 16, 32, 16, 4), bottlenecks (4, 8, 8, 2, 1), plain nearest decoder, dense dilation
# pattern): the network the analytical per-block flow (MILP/analytical/net_fold.py) assembles. Only INPUT_HW and NET_NAME differ from the 512 config; its
# docstring below is kept verbatim for provenance.
"""Shared constants for nnUNetTrainerENet_12_dense_relu_nearest_upsample's
("S12-dense, bare nearest-neighbor decoder") HAWQ per-layer W/A search -- the
already-trained checkpoint (compression/results.csv's own
nnUNetTrainerENet_12_dense_relu_nearest_upsample row, stage
"12_dense_relu_nearest_upsample", dice=0.7734, 140/150 epochs, see
compression/slurm/stage_12_dense_relu_nearest_upsample_warmstart150ep.job's
header for this checkpoint's own training status). 512x512 patch size,
confirmed via this net's own plans.json (`patch_size: [512, 512]`).

Byte-for-byte config_12_dense_relu_nearest_conv_upsample.py's own recipe
(same channels, native bottleneck depth, plain "dense_dilation" context
pattern, no DSC, plain ReLU, SEPARABLE_DILATED=False) -- the ONLY change is
DECODER_TYPE: "nearest_upsample" instead of "nearest_conv_upsample". Per
ENet.py's UpsamplingBottleneck (`enet/nnunetv2/nets/ENet.py:840-897`): both
decoder types do `main_proj` (1x1 conv+BN) -> `F.interpolate(mode='nearest')`
on the skip/main branch, and an identical learned `ConvTranspose2d(k=2,s=2)`
+BN+activation on the residual/expand branch (`reduce` -> `up` -> `expand`,
unrelated to `decoder_type`) -- "nearest_conv_upsample" then additionally
runs the skip branch through a learned 3x3 Conv2d+BN+activation
(`skip_resize_conv`, the classic "resize-convolution" anti-checkerboard
pattern) that "nearest_upsample" omits entirely. So this config's traced
graph has NO `up4.skip_resize_conv.0`/`up5.skip_resize_conv.0` nodes at all
-- every other node (encoder in full, and the rest of up4/up5) is
byte-for-byte the same shape as the `nearest_conv_upsample` family. See
config_23_1.py for the full rationale on why per-decoder-type config files
exist, not repeated here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_upsample_256"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "nearest_upsample"
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

INPUT_HW = (256, 256)  # 256x256 patch; overrides finn_milp.INPUT_HW (512)
