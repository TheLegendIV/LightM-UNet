"""Shared constants for nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256_w8_16's
("S12-dense, nearest+conv decoder, 256x256/Dataset510_ARCADE_256_4c, w8/16
width point") HAWQ per-layer W/A search, once compression/slurm/stage_12_
dense_relu_nearest_conv_upsample_256_w8_16.job has produced a checkpoint
(stage "12_dense_relu_nearest_conv_upsample_256" in compression/results.csv,
config_name suffix "_w8_16").

Byte-for-byte config_12_dense_relu_nearest_conv_upsample_256.py's own recipe
-- same bottleneck depth, decoder, context pattern, flags -- the ONLY change
is CHANNELS: (4, 8, 16, 8, 4) instead of (4, 16, 32, 16, 4) (f_i=4, f1=8,
f2=f3=16, f4=8, f5=4 -- the same width as the 512x512 family's own
nnUNetTrainerENet_12_dense_relu_w8_16, stage 12_dense_relu_width_sweep, see
analysis/report/model_search's Stage 7). Every quantizer site NAME is
identical to the base width (channel width never changes a module's
attribute path, only its tensor shapes), but the layer_bits_*.json search
itself is NOT transferable across widths -- HAWQ sensitivity and the FINN
folding/resource cost model both depend on the actual per-layer tensor
shapes, so this width needs its own ILP solve under MILP/artifacts/, not a
reuse of the base width's S12_dense_nn_upsample_256_v1/v2 artifacts.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256_w8_16"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 8, 16, 8, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "nearest_conv_upsample"
CONTEXT_PATTERN = "dense_dilation_half"
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
INPUT_HW = (256, 256)  # this family trains at 256x256 (Dataset510 plans patch_size); overrides finn_milp.INPUT_HW
