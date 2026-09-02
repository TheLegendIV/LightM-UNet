"""HYPOTHETICAL architecture -- no trained checkpoint exists for this combo
yet. This config exists purely so finn_block_costs.py/folding_ilp.py can
trace real geometry at this exact architecture for a hardware-cost
estimate -- do NOT attempt to run block_sensitivity.py against it, there is
no real checkpoint to load.

w16 width (4,8,16,8,4), plain dense_dilation context (native depth 8, no
reg-interleaved bookends -- "no reg"), plain ReLU (use_prelu=False), and
dsc_no_projection SCOPED TO CONTEXT ONLY (dsc_no_projection_context_only=
True) -- unlike nnUNetTrainerENet_8_2_relu_no_reg_w16 (unscoped
dsc_no_projection=True), regular1/regular4/regular5 here stay REAL
projected RegularBottleneck blocks with real ReLU ("rest is normal ENet
style"); only stage2/stage3's dilated slots become DSCNoProjectionBottleneck.

See compression/hawq/hypothetical_w16_no_reg_context_only.py for how the
per-block bit assignment is approximated (reusing block_bits_8_2_relu_
no_reg_w16_acc2x_min4_joint.json, computed for the UNSCOPED sibling's own
architecture, not this exact one -- no real sensitivity exists for a real
RegularBottleneck+ReLU combination at this width)."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_w16_no_reg_context_only"  # HYPOTHETICAL -- no such checkpoint exists
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
