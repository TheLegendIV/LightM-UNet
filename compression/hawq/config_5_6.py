"""Shared constants for nnUNetTrainerENet_5_6_separable_dense_dilation's
("S5.6") HAWQ per-block W/A search -- the classic, ALREADY-TRAINED S5.6
checkpoint (compression/results.csv's own nnUNetTrainerENet_5_6_separable_
dense_dilation row, stage=5_arch_probe_pairs, dice=0.7985), not the
S19-depth/nonneg_block variant compression/slurm/
stage_s19_dense_dilation_warmstart.job is training separately (that one has
no real checkpoint yet -- block_sensitivity.py's strict=True load needs a
genuinely-trained checkpoint of the EXACT architecture, which this one
already is).

context_pattern=dense_dilation (plain, no reg-bookend -- S5.6's own
defining recipe), prelu_variant=standard (real per-channel PReLU --
confirmed directly against the checkpoint's own state_dict:
`initial.act.weight` shape (4,), `down1.out_act.weight` shape (16,), i.e.
one scalar per CHANNEL, not per-block -- NOT nonneg_block, which config_23_1.py's/
config_26_5_w24.py's own docstrings already establish saves a single (1,)-shaped
scalar per block instead). See config_23_1.py for the full rationale on why
this file exists (one source of truth per architecture, shared across
sensitivity.py/finn_stage_costs.py) and the 5-stage grouping's own
justification -- not repeated here since it's identical.

Depth (4,8,8,2,1) is S5.6's OWN depth, shallower than S19's (4,12,12,2,1) --
do not confuse this with config_26_5_w24.py (S5.6's width-bumped/narrowed
descendant, channels 4,8,24,8,4, SAME depth 4,8,8,2,1) or config_23_1.py/
S19 (S19's OWN depth, different context_pattern)."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_5_6_separable_dense_dilation"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = True
PRELU_VARIANT = "standard"

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
