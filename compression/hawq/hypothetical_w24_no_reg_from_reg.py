"""One-off, NOT a general pipeline script: answers "how much LUT/latency
would nnUNetTrainerENet_8_2_relu_w24_reg save if its 3 reg-bookend blocks
per context stage (RegularBottleneck spacers at slot indices 0/5/10 of the
11-slot dense_dilation_reg_interleaved pattern -- stage2.0/2.5/2.10 and
stage3.0/3.5/3.10) were removed, converting it to the plain 8-slot
dense_dilation pattern (no bookends, just d2/d4/d8/d16 x2)" -- WITHOUT
training a new checkpoint or computing fresh sensitivity for that
hypothetical architecture.

Approximation, explicitly for a back-of-envelope comparison only: reuses
w24_reg's own REAL per-block sensitivity-derived acc1x bit assignment
unchanged for every surviving (non-bookend) block, and simply drops the 6
bookend blocks' own LUT/BRAM/cycle contribution entirely (as if they were
never there) rather than re-running the whole HAWQ search on an actual
no-reg w24 checkpoint (which doesn't exist -- see nnUNetTrainerENet_
8_2_relu_w24_no_reg_d2_projected, the only real trained w24 no-reg sibling,
and it differs structurally at d=2 too, not a clean subset). Good enough to
answer "how big is the reg-bookend tax," not a substitute for a real
trained-and-measured no-reg-w24-plain result.

Usage: python compression/hawq/hypothetical_w24_no_reg_from_reg.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.ENet import ENet  # noqa: E402

from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from folding_ilp import INPUT_HW, XCZU7EV, solve_folding  # noqa: E402
import config_8_2_relu_w24_reg as cfg  # noqa: E402

BOOKEND_BLOCKS = {"stage2.0", "stage2.5", "stage2.10", "stage3.0", "stage3.5", "stage3.10"}

model = ENet(
    in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=cfg.CHANNELS,
    bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE, decoder_type=cfg.DECODER_TYPE,
    use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN,
    separable_dilated=cfg.SEPARABLE_DILATED, use_prelu=cfg.USE_PRELU, prelu_variant=cfg.PRELU_VARIANT,
    use_dsc=cfg.USE_DSC, dsc_no_projection=cfg.DSC_NO_PROJECTION,
    dsc_no_projection_context_only=cfg.DSC_NO_PROJECTION_CONTEXT_ONLY, reg_bookend_dsc=cfg.REG_BOOKEND_DSC,
)
geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
print(f"Full w24_reg: {len(geometries)} layers across {len(block_names)} blocks.")

no_reg_geometries = [g for g in geometries if g.stage not in BOOKEND_BLOCKS]
removed = len(geometries) - len(no_reg_geometries)
print(f"Dropping {removed} layers belonging to the {len(BOOKEND_BLOCKS)} reg-bookend blocks: {sorted(BOOKEND_BLOCKS)}")
print(f"Hypothetical no-reg: {len(no_reg_geometries)} layers across {len(block_names) - len(BOOKEND_BLOCKS)} blocks.")

with open(REPO_ROOT / "compression/hawq/artifacts/block_bits_8_2_relu_w24_reg_acc1x_joint.json") as f:
    bits = json.load(f)
stage_bits = {"stage_weight_bits": bits["stage_weight_bits"], "stage_act_bits": bits["stage_act_bits"]}

for label, hard_lut, hard_bram, lut_weight, bram_weight in [
    ("balanced (default penalty, unconstrained)", None, None, 1.0, 1.0),
    ("min-latency, hard-capped at 100% LUT", 1.0, 1.0, 0.0, 0.0),
    ("min-latency, hard-capped at 150% LUT (BRAM uncapped)", 1.5, None, 0.0, 0.0),
    ("min-latency, fully unconstrained", None, None, 0.0, 0.0),
]:
    result = solve_folding(no_reg_geometries, stage_bits, 8, 8, lut_weight, bram_weight, hard_lut=hard_lut, hard_bram=hard_bram)
    diag = result["_diagnostics"]
    cycles = diag["total_cycles"]
    print(f"\n[{label}]")
    print(f"  status: {result['status']}")
    print(f"  LUT: {diag['lut_pct_of_budget']:.1f}% of budget ({diag['total_lut_calibrated']:.0f} calibrated)")
    print(f"  BRAM_18K: {diag['bram_pct_of_budget']:.1f}% of budget ({diag['total_bram18k_calibrated']:.0f} calibrated)")
    print(f"  cycles: {cycles:.0f}  (~{cycles / 100e6 * 1000:.1f} ms @ 100MHz)")
