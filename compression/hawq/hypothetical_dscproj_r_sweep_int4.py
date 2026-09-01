"""One-off, NOT a general pipeline script: hardware-cost estimate (uniform
INT4, W4A4) for the new S8.2 "no reg" DSC-with-projection architecture
(RegularBottleneck's own pre-existing use_dsc=True branch: real 1x1
reduce/expand around the depthwise+pointwise DSC pair, at internal ratio r
-- ENet.py's new dsc_internal_ratio constructor param, added this session),
swept at r in {2,4,8}, channels=(4,16,32,16,4) (native S8.2 width, same as
nnUNetTrainerENet_8_2_relu_no_reg_fullwidth's own width). Purely a geometry
+ FINN-cost-model estimate -- NO checkpoint needed (no real sensitivity,
just "what would uniform int4 cost on this architecture's own conv
geometry"), since none of these 3 configs have been trained yet (the real
FP32 array job -- stage_8_2_relu_no_reg_dscproj_r_array.job -- hasn't run
yet at the time this script was written).

Usage: python compression/hawq/hypothetical_dscproj_r_sweep_int4.py
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

for r in (2, 4, 8):
    model = ENet(
        in_channels=1, out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        decoder_type="upsample_conv", use_asymmetric=False, context_pattern="dense_dilation",
        separable_dilated=False, use_prelu=False, use_dsc=True, dsc_no_projection=False, dsc_internal_ratio=r,
    )
    geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n===== r={r} (internal width = C/{r}) =====")
    print(f"  {len(geometries)} layers, {len(block_names)} blocks, {n_params} params")

    for scenario, hard_lut, hard_bram, lut_weight, bram_weight in [
        ("balanced (default penalty, unconstrained)", None, None, 1.0, 1.0),
        ("min-latency, hard-capped at 100% LUT", 1.0, 1.0, 0.0, 0.0),
    ]:
        result = solve_folding(geometries, None, 4, 4, lut_weight, bram_weight, hard_lut=hard_lut, hard_bram=hard_bram)
        diag = result["_diagnostics"]
        cycles = diag["total_cycles"]
        print(f"  [{scenario}] status={result['status']}  "
              f"LUT={diag['lut_pct_of_budget']:.1f}%  BRAM={diag['bram_pct_of_budget']:.1f}%  "
              f"cycles={cycles:.0f} (~{cycles / 100e6 * 1000:.1f} ms @ 100MHz)")
