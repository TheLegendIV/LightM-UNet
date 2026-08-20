"""Shared constants for nnUNetTrainerENet_23_1_s19_warmstart_4c's HAWQ
per-stage W/A search (compression/hawq/sensitivity.py, finn_stage_costs.py,
ilp_search.py) -- one source of truth for the architecture config and the
5-stage grouping, instead of duplicating it into each script (which is what
enet/nnunetv2/nets/QuantENet23_1.py's own module-level constants already
are for the Brevitas side -- this mirrors them for the plain-FP32/analysis
side, which has no brevitas dependency and shouldn't need one just to read
these numbers).

Architecture confirmed from compression/slurm/stage_23_1_s19_warmstart_4c.job
and compression/results.csv (byte-identical to its warm-start source, S19).
Stage grouping confirmed against hardware/finn_stage_partition.py's own
5-way partition boundaries (built independently for Vivado stitching, same
initial/stage1/stage2+3/stage4/stage5 split)."""
from __future__ import annotations

IN_CHANNELS = 1
OUT_CHANNELS = 5  # labels: background, LAD, RCA, LCX, LM
CHANNELS = (4, 16, 32, 16, 4)  # initial, stage1, stage2/3 (context), stage4, stage5
BOTTLENECKS_PER_STAGE = (4, 12, 12, 2, 1)
DECODER_TYPE = "upsample_conv"
CONTEXT_PATTERN = "dense_dilation_reg_interleaved_double_mid"
USE_ASYMMETRIC = False
SEPARABLE_DILATED = True
PRELU_VARIANT = "nonneg_block"

# Top-level ENet.py/QuantENet23_1.py attribute names belonging to each of
# the 5 HAWQ stages (matches ENET_CHANNELS/ENET_BOTTLENECKS convention and
# hardware/finn_stage_partition.py's own boundary rule: cut right after
# each down/upsampling bottleneck's first op).
STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}
# The boundary activation tensor representing each stage's OUTPUT (used by
# sensitivity.py's activation-trace hooks).
STAGE_BOUNDARY_ATTR = {
    "initial": "initial",
    "stage1": "regular1",
    "context": "stage3",
    "stage4": "regular4",
    "stage5": "regular5",  # self.final has no activation quantizer of its own
}
STAGE_NAMES = tuple(STAGE_MODULE_ATTRS.keys())
CANDIDATE_BITS = (2, 4, 8)
