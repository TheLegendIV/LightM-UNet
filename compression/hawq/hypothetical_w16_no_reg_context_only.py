"""One-off, NOT a general pipeline script: hardware-cost/latency estimate
for the HYPOTHETICAL "w16, no reg, context-only dsc_no_projection, ReLU"
architecture (config_8_2_relu_w16_no_reg_context_only.py) -- no trained
checkpoint exists for this combination, so there is no real per-block
sensitivity data to run the real HAWQ search against.

Bit-assignment approximation, explicitly for a back-of-envelope estimate
only: down1/down2/stage2.*/stage3.*/up4/up5/initial/final are structurally
IDENTICAL modules (DSCNoProjectionBottleneck / plain conv) to
nnUNetTrainerENet_8_2_relu_no_reg_w16's own equivalents (context ALWAYS
gets dsc_no_projection regardless of context-only scoping -- only whether
OUTSIDE-context stages ALSO get it differs) -- so those blocks reuse THAT
net's real per-block acc2x (4/8-bit-only ILP) bit assignment
(block_bits_8_2_relu_no_reg_w16_acc2x_min4_joint.json) unchanged.
regular1.0-3/regular4.0-1/regular5.0 are a DIFFERENT block type here (real
projected RegularBottleneck, "rest is normal ENet style") than in the
unscoped sibling (DSCNoProjectionBottleneck there) -- no real sensitivity
exists anywhere for that combination, so those blocks fall back to a flat,
conservative W8A8.

Usage: python compression/hawq/hypothetical_w16_no_reg_context_only.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.ENet import ENet  # noqa: E402

from block_utils import enumerate_blocks  # noqa: E402
from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from folding_ilp import INPUT_HW, solve_folding  # noqa: E402
import config_8_2_relu_w16_no_reg_context_only as cfg  # noqa: E402

REUSED_BLOCK_PREFIXES = ("initial", "down1", "down2", "stage2", "stage3", "up4", "up5", "final")
FALLBACK_W8A8_PREFIXES = ("regular1", "regular4", "regular5")

model = ENet(
    in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=cfg.CHANNELS,
    bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE, decoder_type=cfg.DECODER_TYPE,
    use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN,
    separable_dilated=cfg.SEPARABLE_DILATED, use_prelu=cfg.USE_PRELU, prelu_variant=cfg.PRELU_VARIANT,
    use_dsc=cfg.USE_DSC, dsc_no_projection=cfg.DSC_NO_PROJECTION,
    dsc_no_projection_context_only=cfg.DSC_NO_PROJECTION_CONTEXT_ONLY, reg_bookend_dsc=cfg.REG_BOOKEND_DSC,
)
blocks = enumerate_blocks(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"Enumerated {len(blocks)} blocks, {n_params} params:")
for name, module in blocks.items():
    print(f"  {name}: {type(module).__name__}")

with open(REPO_ROOT / "compression/hawq/artifacts/block_bits_8_2_relu_no_reg_w16_acc2x_min4_joint.json") as f:
    donor_bits = json.load(f)

stage_weight_bits, stage_act_bits = {}, {}
for name in blocks:
    prefix = name.split(".")[0]
    if prefix in REUSED_BLOCK_PREFIXES:
        stage_weight_bits[name] = donor_bits["stage_weight_bits"][name]
        stage_act_bits[name] = donor_bits["stage_act_bits"][name]
    elif prefix in FALLBACK_W8A8_PREFIXES:
        stage_weight_bits[name] = 8
        stage_act_bits[name] = 8
    else:
        raise ValueError(f"Unhandled block prefix for {name!r} -- add it to REUSED_BLOCK_PREFIXES or FALLBACK_W8A8_PREFIXES.")
stage_bits = {"stage_weight_bits": stage_weight_bits, "stage_act_bits": stage_act_bits}
print(f"\nBit assignment: {sum(1 for n in blocks if n.split('.')[0] in REUSED_BLOCK_PREFIXES)} blocks reused from "
      f"w16's real acc2x_min4 bits, {sum(1 for n in blocks if n.split('.')[0] in FALLBACK_W8A8_PREFIXES)} "
      f"blocks (regular1/regular4/regular5, real RegularBottleneck -- no analog available) fall back to flat W8A8.")

geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
print(f"\n{len(geometries)} layers across {len(block_names)} blocks.")

for scenario, hard_lut, hard_bram, lut_weight, bram_weight in [
    ("balanced (default penalty, unconstrained)", None, None, 1.0, 1.0),
    ("min-latency, hard-capped at 100% LUT", 1.0, 1.0, 0.0, 0.0),
]:
    result = solve_folding(geometries, stage_bits, 8, 8, lut_weight, bram_weight, hard_lut=hard_lut, hard_bram=hard_bram)
    diag = result["_diagnostics"]
    cycles = diag["total_cycles"]
    print(f"  [{scenario}] status={result['status']}  "
          f"LUT={diag['lut_pct_of_budget']:.1f}%  BRAM={diag['bram_pct_of_budget']:.1f}%  "
          f"cycles={cycles:.0f} (~{cycles / 100e6 * 1000:.1f} ms @ 100MHz)")
