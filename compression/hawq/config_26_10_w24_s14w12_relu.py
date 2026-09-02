"""Shared constants for nnUNetTrainerENet_26_10_w24_s14w12_relu's HAWQ
per-block search (compression/results.csv's own 26_s5_6_probe_family row,
dice=0.7360). Channels (4,12,24,12,4), plain dense_dilation context (native
depth 8), plain ReLU (use_prelu=False). Unlike the whole S8.2 family, this
net does NOT use dsc_no_projection/use_dsc at all -- its own dilated-conv
factorization is separable_dilated=True (a DENSE (k,1)+(1,k) two-pass
factoring of the dilated conv, groups=1 throughout -- NOT grouped/depthwise
convs, so the groups-aware finn_cost_model.py fix from earlier this session
is a no-op for this architecture, per that fix's own module comment).

See config_8_2_relu_w24_no_reg_d2_projected.py for the full USE_PRELU/
DSC_NO_PROJECTION-as-optional-config-globals rationale -- not repeated
here (this config simply never sets USE_DSC/DSC_NO_PROJECTION, so those
call sites fall back to ENet.py's own defaults of False either way)."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_26_10_w24_s14w12_relu"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 12, 24, 12, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = True
PRELU_VARIANT = "standard"  # unused: USE_PRELU=False collapses the whole encoder to plain ReLU regardless
USE_PRELU = False

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
