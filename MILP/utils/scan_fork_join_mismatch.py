"""Scans the CURRENT/deployed per-layer folding result (a finn_milp.py
--out-file JSON) for fork/join rate mismatches: every point where two (or
more) branches reconverge (a residual add/concat), using layer_topology.
compute_predecessor_map to find the join sites and each branch's nearest
real (Conv2d/ConvTranspose2d/MaxPool2d) layer, then compares those layers'
OWN "cycles" (already computed by finn_milp.py's cost model, at the SAME
resolution on both branches since add/concat requires matching shapes) to
rank every join by how imbalanced its two incoming branches are.

This does NOT need FINN/HLS/rtlsim at all -- "cycles" is exactly the same
per-layer closed-form cycle estimate finn_milp.py's own z/raw_cycles
already uses, just re-consumed from a SOLVED result to see whether the
current folding choice leaves any real fork/join imbalance behind.

Usage (host Python, same env finn_milp.py itself runs in):
    python MILP/scan_fork_join_mismatch.py \\
        --config config_12_dense_relu_warmstart150ep \\
        --folding-file MILP/artifacts/12_dense_relu_warmstart150ep_ILP_outputs_v3/layer_bits_folding_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut50_bram70_dsp90.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401 -- side-effect parity with finn_milp.py's own import order

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_milp import INPUT_HW, load_config, trace_layer_geometry  # noqa: E402
from layer_topology import compute_predecessor_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="MILP/config_*.py module name (no .py).")
    parser.add_argument("--folding-file", type=Path, required=True,
                         help="finn_milp.py --out-file JSON (has a top-level 'per_layer' dict with a "
                              "'cycles' entry per layer name).")
    parser.add_argument("--top", type=int, default=None, help="Only print the N worst joins (default: all).")
    args = parser.parse_args()

    load_config(args.config)
    from finn_milp import IN_CHANNELS, CHANNELS, BOTTLENECKS_PER_STAGE, DECODER_TYPE, CONTEXT_PATTERN  # noqa: E402
    import finn_milp as fm

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=fm.OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=fm.USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=fm.SEPARABLE_DILATED, use_prelu=fm.__dict__.get("USE_PRELU", True),
        prelu_variant=fm.PRELU_VARIANT, use_dsc=fm.__dict__.get("USE_DSC", False),
        dsc_no_projection=fm.__dict__.get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=fm.__dict__.get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=fm.__dict__.get("REG_BOOKEND_DSC", False),
        dsc_separable=fm.__dict__.get("DSC_SEPARABLE", False),
    )
    geometries, _ = trace_layer_geometry(model, INPUT_HW, IN_CHANNELS)
    geom_by_name = {g.name: g for g in geometries}

    predecessor_map = compute_predecessor_map(model)

    with open(args.folding_file) as f:
        folding = json.load(f)
    per_layer = folding["per_layer"]

    rows = []
    for join_name, preds in predecessor_map.items():
        if len(preds) < 2:
            continue
        branch_cycles = []
        for p in preds:
            if p not in per_layer:
                branch_cycles.append((p, None))
                continue
            branch_cycles.append((p, per_layer[p]["cycles"]))
        known = [(p, c) for p, c in branch_cycles if c is not None]
        if len(known) < 2:
            continue
        fastest = min(known, key=lambda pc: pc[1])
        slowest = max(known, key=lambda pc: pc[1])
        ratio = slowest[1] / fastest[1] if fastest[1] > 0 else float("inf")
        g = geom_by_name.get(join_name)
        n_pixels = g.hin * g.win if g is not None else None
        predicted_depth = n_pixels * (1 - fastest[1] / slowest[1]) if n_pixels else None
        rows.append({
            "join": join_name, "branches": branch_cycles, "fastest": fastest, "slowest": slowest,
            "ratio": ratio, "n_pixels": n_pixels, "predicted_depth": predicted_depth,
        })

    rows.sort(key=lambda r: r["ratio"], reverse=True)
    if args.top:
        rows = rows[: args.top]

    print(f"{'join layer':<28} {'ratio':>8} {'pred_depth':>11} {'fastest (cycles)':<38} {'slowest (cycles)':<38}")
    for r in rows:
        fastest_str = f"{r['fastest'][0]}={r['fastest'][1]:.0f}"
        slowest_str = f"{r['slowest'][0]}={r['slowest'][1]:.0f}"
        depth_str = f"{r['predicted_depth']:.0f}" if r["predicted_depth"] is not None else "?"
        print(f"{r['join']:<28} {r['ratio']:>8.2f} {depth_str:>11} {fastest_str:<38} {slowest_str:<38}")

    n_severe = sum(1 for r in rows if r["ratio"] >= 2.0)
    print(f"\n{len(rows)} fork/join point(s) found; {n_severe} with ratio >= 2x "
          f"(the current folding's own real imbalances, not just the initial block).")


if __name__ == "__main__":
    main()
