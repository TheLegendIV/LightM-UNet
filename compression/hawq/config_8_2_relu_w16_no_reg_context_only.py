"""w16 width (4,8,16,8,4), plain dense_dilation context (native depth 8, no
reg-interleaved bookends -- "no reg"), plain ReLU (use_prelu=False), and
dsc_no_projection SCOPED TO CONTEXT ONLY (dsc_no_projection_context_only=
True) -- unlike nnUNetTrainerENet_8_2_relu_no_reg_w16 (unscoped
dsc_no_projection=True), regular1/regular4/regular5 here stay REAL
projected RegularBottleneck blocks with real ReLU ("rest is normal ENet
style"); only stage2/stage3's dilated slots become DSCNoProjectionBottleneck.

UPDATE: a real checkpoint now exists (trained to a complete 150-epoch
schedule, dice=0.7282, see compression/results.csv's own
8_2_relu_w16_no_reg_context_only row) -- block_sensitivity.py CAN be run
against this config directly now. The note below about
hypothetical_w16_no_reg_context_only.py's fallback-bits approximation is
kept for historical context only; prefer a real
block_sensitivity_8_2_relu_w16_no_reg_context_only.json search over that
flat-W8A8-fallback approximation whenever both exist.

See compression/hawq/hypothetical_w16_no_reg_context_only.py for how the
per-block bit assignment was PREVIOUSLY approximated, before a real
checkpoint existed (reusing block_bits_8_2_relu_no_reg_w16_acc2x_min4_
joint.json, computed for the UNSCOPED sibling's own architecture, not this
exact one, plus a flat W8A8 fallback for the 7 real RegularBottleneck
blocks that sibling doesn't have)."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_w16_no_reg_context_only"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 8, 16, 8, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = False
PRELU_VARIANT = "standard"  # unused: USE_PRELU=False collapses the whole encoder to plain ReLU regardless
USE_PRELU = False
USE_DSC = False
DSC_NO_PROJECTION = True
DSC_NO_PROJECTION_CONTEXT_ONLY = True
REG_BOOKEND_DSC = False

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
