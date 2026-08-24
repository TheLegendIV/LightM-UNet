"""Shared constants for nnUNetTrainerENet_26_5_w24's HAWQ per-stage W/A
search -- S5-SeparableDense (5_6_separable_dense_dilation) with stage2/3
width bumped 32 -> 24 (channels=4,16,32,16,4 -> 4,8,24,8,4, see
compression/slurm/stage_26_s5_6_probe_family_array.job's own header comment
and compression/finn_cost_s5_6_variants.py's cost screening), confirmed
against compression/results.csv's nnUNetTrainerENet_26_5_w24 row, stage
26_s5_6_probe_family (dice=0.7566, params=12401).

Unlike config_23_1.py/config_21_2.py (S19/23_1 family: context_pattern=
dense_dilation_reg_interleaved_double_mid, prelu_variant=nonneg_block), this
is S5.6's OWN recipe: context_pattern=dense_dilation (plain, no reg-bookend),
prelu_variant=standard (real per-channel PReLU, not the FINN-deployable
nonneg_block scalar -- S5.6's family was never retrained with nonneg_block).

See config_23_1.py for the full rationale on why this file exists (one
source of truth per architecture, shared across sensitivity.py/
finn_stage_costs.py) and the 5-stage grouping's own justification -- not
repeated here since it's identical."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_26_5_w24"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 8, 24, 8, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
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
