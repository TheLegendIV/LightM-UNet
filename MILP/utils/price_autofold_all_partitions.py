"""Price FINN's auto-fold for the WHOLE network with finn_cost_model.py and print the result as
finn_milp.py hard-cap flags (fractions of the xczu7ev board) plus the achieved FPS.

Reads, per partition N in 0..7, from --dir:
    autofold_config_partitionN.json   (PE/SIMD per node, from dump_autofold_config_all_partitions.py)
    autofold_partitionN.onnx          (the same run's auto-folded partition graph: geometry/datatypes)
and sums apply_folding_config_cost.estimate() over partitions. Fails if any partition is missing
unless --partial is given (then the totals are labelled PARTIAL and must not be used as MILP caps).

Usage (needs only `onnx`; lightmunet_dev container or host Python):
    python MILP/utils/price_autofold_all_partitions.py \
        --dir hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/autofold_configs_all_partitions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_folding_config_cost import XCZU7EV, estimate  # noqa: E402

N_PARTITIONS = 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--partial", action="store_true", help="Skip partitions with a missing .onnx (totals are PARTIAL).")
    ap.add_argument("--onnx-override", action="append", default=[], metavar="N=PATH",
                    help="Use PATH as partition N's graph (e.g. 2=<existing partition2 checkpoint>).")
    args = ap.parse_args()

    overrides = {int(k): Path(v) for k, v in (o.split("=", 1) for o in args.onnx_override)}
    missing, per_part = [], []
    for n in range(N_PARTITIONS):
        cfg = args.dir / f"autofold_config_partition{n}.json"
        graph = overrides.get(n, args.dir / f"autofold_partition{n}.onnx")
        if not cfg.is_file() or not graph.is_file():
            missing.append(n)
            continue
        per_part.append((n, estimate(graph, cfg, True, args.clock_mhz)))
    if missing and not args.partial:
        print(f"ERROR: partition(s) {missing} missing config or .onnx in {args.dir} (use --partial to skip).")
        return 1

    tot = {k: sum(r[k] for _, r in per_part) for k in ("total_lut", "total_bram18", "total_uram18", "total_dsp")}
    slowest = max((r for _, r in per_part), key=lambda r: r["bottleneck_cycles"])
    print(f"{'part':>4s} {'LUT':>9s} {'BRAM18':>8s} {'DSP':>6s} {'bottleneck':<28s} {'cycles':>10s} {'FPS':>8s}")
    for n, r in per_part:
        print(f"{n:>4d} {r['total_lut']:>9.0f} {r['total_bram18']:>8.1f} {r['total_dsp']:>6.0f} "
              f"{r['bottleneck_name']:<28s} {r['bottleneck_cycles']:>10.0f} {r['fps']:>8.1f}")
    label = f"PARTIAL (missing partitions {missing})" if missing else "WHOLE NETWORK"
    print(f"\n{label}: LUT {tot['total_lut']:.0f}  BRAM18 {tot['total_bram18']:.1f}  DSP {tot['total_dsp']:.0f}  "
          f"URAM {tot['total_uram18']:.1f}")
    print(f"Slowest node overall: {slowest['bottleneck_name']} at {slowest['bottleneck_cycles']:.0f} cycles "
          f"-> {slowest['fps']:.2f} FPS @ {args.clock_mhz:.0f} MHz\n")
    print(f"# finn_milp.py flags equal to FINN auto-fold's cost ({label}):")
    print(f"--hard-lut-fraction {tot['total_lut'] / XCZU7EV['LUT']:.4f}")
    print(f"--hard-bram-fraction {tot['total_bram18'] / XCZU7EV['BRAM_18K']:.4f}")
    print(f"--hard-dsp-fraction {tot['total_dsp'] / XCZU7EV['DSP']:.4f}")
    print(f"--hard-uram-fraction {tot['total_uram18'] / XCZU7EV['URAM']:.4f}")
    print(f"--target-fps {slowest['fps']:.1f}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
