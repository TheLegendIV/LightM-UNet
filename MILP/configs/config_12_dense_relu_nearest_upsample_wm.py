"""Shared constants for nnUNetTrainerENet_12_dense_relu_nearest_upsample_warmstart150ep's
("S12-dense, bare nearest-neighbor decoder, WARM-STARTED continuation") HAWQ
per-layer W/A search -- the already-trained checkpoint (compression/results.csv's
own nnUNetTrainerENet_12_dense_relu_nearest_upsample_warmstart150ep row, stage
"12_dense_relu_nearest_upsample_warmstart150ep", dice=0.7863, 149/150 epochs --
an improvement over the base `..._nearest_upsample` checkpoint's own
dice=0.7734 (see config_12_dense_relu_nearest_upsample.py), from a second,
reduced-LR 150-epoch schedule warm-started from that checkpoint, see
compression/slurm/stage_12_dense_relu_nearest_upsample_warmstart150ep.job).
Same 512x512 patch size, same architecture shape entirely -- this checkpoint
differs from the base one ONLY in trained weights (continued training), not
in any architecture parameter below.

"_wm" is this config's own naming shorthand for "warmstart150ep" throughout
MILP/hardware/QAT artifacts for this specific checkpoint (e.g.
`S12_dense_nearest_upsample_wm_v1`), to keep filenames shorter than spelling
out "warmstart150ep" everywhere.

Byte-for-byte config_12_dense_relu_nearest_upsample.py's own recipe (same
channels, bottleneck depth, context pattern, decoder type) -- the ONLY
difference between the two config files is NET_NAME (which checkpoint's
weights to load for HAWQ sensitivity measurement). See that file's own
docstring for the full decoder_type="nearest_upsample" rationale (bare
nearest-neighbor resize, no skip_resize_conv at all), not repeated here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_upsample_warmstart150ep"
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
