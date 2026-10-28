"""Shared constants for nnUNetTrainerENet_12_separable_dense_relu_warmstart150ep's
("S12.1-separable, warm-started") HAWQ per-layer W/A search -- the WARM-
STARTED checkpoint (compression/slurm/stage_12_separable_dense_relu_warmstart150ep.job:
a fresh 150-epoch continued-training run warm-started from
nnUNetTrainerENet_12_separable_dense_relu's own checkpoint_best.pth, dice=
0.7729 @ 144 epochs, at a reduced LR=1e-4 -- byte-for-byte the same warm-start
treatment already given to 12_dense_relu -> 12_dense_relu_warmstart150ep, see
that pair's own config_12_dense_relu_warmstart150ep.py for the full
rationale).

Byte-for-byte config_12_separable_dense_relu.py's own architecture (same
channels, native bottleneck depth, plain "dense_dilation" context pattern,
no DSC, plain ReLU, SEPARABLE_DILATED=True -- dense conv factored into a
(k,1)+(1,k) pair) -- only NET_NAME differs, pointing at the warm-started
checkpoint instead of the original 144-epoch one. See config_23_1.py for the
full rationale on why this file exists -- not repeated here.
"""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_12_separable_dense_relu_warmstart150ep"
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
