"""Shared constants for nnUNetTrainerENet_1_naive_baseline_U8's HAWQ
per-stage W/A search / folding ILP -- the "U8" point of stage
1_naive_baseline's channel-width grid (config_abbreviations.csv's S1.6:
CHANNELS=4,8,16,8,4, BOTTLENECKS=4,8,8,2,1, decoder_type=upsample_conv,
use_prelu=0, use_dilated=1, use_asymmetric=1, use_strided=1, use_dsc=0,
context_pattern=default -- see compression/slurm/stage_1_naive_baseline_
array.job's own ENET_* env vars, the authoritative source for this stage).
use_dilated/use_strided/use_dsc all match ENet's own constructor defaults
(True/True/False) so they don't need an explicit constant here, same as
config_23_1.py/config_21_2.py not overriding them either.

use_prelu=0 for the real trained U8 checkpoint (plain ReLU, per that slurm
job) -- irrelevant for THIS module's purpose (folding_ilp.py/
finn_stage_costs.py both hardcode use_prelu=True when tracing layer
geometry for cost estimation, since Conv2d/ConvTranspose2d/MaxPool2d
shapes never depend on activation choice). PRELU_VARIANT is kept at the
default "standard" purely for interface consistency with config_23_1.py/
config_21_2.py.

STAGE_MODULE_ATTRS/STAGE_BOUNDARY_ATTR are IDENTICAL to config_23_1.py's --
ENet's top-level submodule attribute names (initial/down1/regular1/down2/
stage2/stage3/up4/regular4/up5/regular5/final) never change with
context_pattern (context_pattern only alters what's built INSIDE
stage2/stage3, not their attribute names) -- confirmed via direct read of
ENet.__init__."""
from __future__ import annotations

NET_NAME = "nnUNetTrainerENet_1_naive_baseline_U8"
IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 8, 16, 8, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "default"
USE_ASYMMETRIC = True
SEPARABLE_DILATED = False
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
