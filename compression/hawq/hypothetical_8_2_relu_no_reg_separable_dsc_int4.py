"""One-off, NOT a general pipeline script: hardware-cost estimate (uniform
INT4, W4A4) for the new S8.2 "no reg" architecture with dsc_separable=True
(ENet.py Stage 18 -- factors DSCNoProjectionBottleneck's own depthwise KxK
pass into a (K,1)+(1,K) depthwise PAIR, a second/independent factoring axis
on top of DSC's existing channel-wise depthwise+pointwise one), same
channels=(4,16,32,16,4)/bottlenecks_per_stage=(4,8,8,2,1)/context_pattern=
"dense_dilation" recipe as nnUNetTrainerENet_8_2_relu_no_reg_fullwidth (see
compression/hawq/config_8_2_relu_no_reg_separable_dsc.py). Purely a
geometry + FINN-cost-model estimate -- NO checkpoint needed (no real
sensitivity, just "what would uniform int4 cost on this architecture's own
conv geometry"), matching hypothetical_dscproj_r_sweep_int4.py's own
precedent (this architecture hasn't been trained yet).

Usage: python compression/hawq/hypothetical_8_2_relu_no_reg_separable_dsc_int4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.ENet import ENet  # noqa: E402

from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from folding_ilp import INPUT_HW, solve_folding  # noqa: E402

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)

for label, dsc_separable in [("dsc_separable=False (baseline, == fullwidth)", False), ("dsc_separable=True (new)", True)]:
    model = ENet(
        in_channels=1, out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        decoder_type="upsample_conv", use_asymmetric=False, context_pattern="dense_dilation",
        separable_dilated=False, use_prelu=False, use_dsc=False, dsc_no_projection=True,
        dsc_separable=dsc_separable,
    )
    geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n===== {label} =====")
    print(f"  {len(geometries)} layers, {len(block_names)} blocks, {n_params} params")

    for scenario, hard_lut, hard_bram, lut_weight, bram_weight in [
        ("balanced (default penalty, unconstrained)", None, None, 1.0, 1.0),
        ("min-latency, hard-capped at 100% LUT+BRAM", 1.0, 1.0, 0.0, 0.0),
    ]:
        result = solve_folding(geometries, None, 4, 4, lut_weight, bram_weight, hard_lut=hard_lut, hard_bram=hard_bram)
        diag = result["_diagnostics"]
        cycles = diag["total_cycles"]
        print(f"  [{scenario}] status={result['status']}  "
              f"LUT={diag['lut_pct_of_budget']:.1f}%  BRAM={diag['bram_pct_of_budget']:.1f}%  "
              f"cycles={cycles:.0f} (~{cycles / 100e6 * 1000:.1f} ms @ 100MHz)")
