"""Shared constants for nnUNetTrainerENet_8_2_relu_w24_no_reg_d2_projected's
HAWQ per-stage search (compression/results.csv's own
8_2_relu_w24_projected_slots_ablation row, dice=0.768). S8.2's own defining
traits (ReLU instead of PReLU, dsc_no_projection=1 unscoped) carried onto
S26_9's own widened context (channels 4,12,24,12,4), but with S8.2's own
context_pattern=dense_dilation_reg_interleaved dropped in favor of S5.6's
plain context_pattern=dense_dilation (native depth 8) -- except the d=2
dilation rate itself is a real, DILATED projected RegularBottleneck instead
of a DSCNoProjectionBottleneck (ENet.py's new DENSE_DILATION_D2_PROJECTED_
PATTERN/dense_dilation_d2_projected context_pattern), restoring genuine
cross-channel mixing at that one rate while d=4/d=8/d=16 stay
DSCNoProjectionBottleneck (no proj + DSC).

USE_PRELU=False and DSC_NO_PROJECTION=True are NOT representable by any
existing config_*.py in this directory -- sensitivity.py's build_fp32_model
and finn_stage_costs.py's/finn_block_costs.py's own ENet(...) construction
used to hardcode use_prelu=True with no dsc_no_projection passthrough at
all (every existing config_*.py this repo has used a PReLU variant with
plain projected bottlenecks, so this gap was never exercised). Extended
those three call sites to read USE_PRELU/DSC_NO_PROJECTION/
DSC_NO_PROJECTION_CONTEXT_ONLY/REG_BOOKEND_DSC/USE_DSC as OPTIONAL config
globals (globals().get(..., <ENet.py's own default>)) -- every existing
config_*.py never defines them, so this is byte-for-byte backward
compatible; this file is the first to actually set USE_PRELU=False and
DSC_NO_PROJECTION=True.

See config_5_6.py for the general rationale on why this per-architecture
config-module pattern exists (one source of truth shared across
sensitivity.py/finn_stage_costs.py/ilp_search.py) -- not repeated here."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_8_2_relu_w24_no_reg_d2_projected"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 12, 24, 12, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_d2_projected"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = False
PRELU_VARIANT = "standard"  # unused: USE_PRELU=False collapses the whole encoder to plain ReLU regardless (see ENet.py's own validation)
USE_PRELU = False
USE_DSC = False
DSC_NO_PROJECTION = True
DSC_NO_PROJECTION_CONTEXT_ONLY = False
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
