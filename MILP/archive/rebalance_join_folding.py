"""Post-hoc, LOCAL repair for a join-branch cycle mismatch in an already-
SOLVED finn_milp.py folding result -- no re-solve of the joint ILP needed.

WHY THIS EXISTS: finn_milp.py folds every layer independently (see its own
module docstring / --max-join-imbalance-ratio's removal) -- nothing during
solving compares two layers that feed the same residual add/concat. For
decoder_type="nearest_conv_upsample" specifically, this leaves a large,
measured mismatch at up4/up5's own join: the new skip_resize_conv.0 (skip/
main branch, nearest resize -> 3x3 conv) vs. expand.0 (residual branch,
reduce->up->expand) -- 6.0x at up4, 3.0x at up5 in the S12_dense_nn_upsample_
v1 alpha=1.0/maxlat150ms solve (786432 vs 131072 cycles, and 393216 vs
131072). A mismatch this size is exactly what forces a real, UNPRICED FIFO
in actual FINN hardware (see finn_cost_model.py's own "still not covered:
FIFOs" note and MILP/scan_fork_join_mismatch.py's `predicted_depth`).

APPROACH: for each (slow_layer, fast_layer) join pair, if slow/fast exceeds
--target-ratio, search the SLOW layer's own candidate_folds (same folding
domain finn_milp.py itself uses) for the CHEAPEST fold -- by extra LUT
(the typically-scarcer resource for this decoder's own S12_dense_nn_
upsample_v1 solve, see --cost-metric) -- that brings its cycles within
target_ratio of the fast layer's (already fixed) cycles. Weight_bits/
act_bits are NOT touched (those came from the ILP's own accuracy-aware
sensitivity term, which this script has no signal to override) -- only the
fold (PE, SIMD, ram_style, variant) changes. Recomputes that one layer's
own LUT/BRAM/DSP/cycles contribution and re-derives the plan's totals under
the SAME calibration this repo's other tools use, so the patched plan's own
new resource usage is reported honestly (NOT re-verified against the hard
budget fractions originally passed to finn_milp.py -- pass --hard-lut-
fraction etc. if you want that checked here too).

Usage:
    python MILP/rebalance_join_folding.py \\
        --config config_12_dense_relu_nearest_conv_upsample \\
        --folding-file MILP/artifacts/S12_dense_nn_upsample_v1/layer_bits_folding_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_forcedsp_lut50_bram30_dsp90_maxlat150ms.json \\
        --target-ratio 1.5 --force-dsp \\
        --out-file MILP/artifacts/S12_dense_nn_upsample_v1/layer_bits_folding_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_forcedsp_lut50_bram30_dsp90_maxlat150ms_joinbalanced.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401 -- side-effect parity with finn_milp.py's own import order

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_cost_model import RAM_STYLE_BLOCK, calibrated_bram18k, calibrated_lut, layer_cost_pe_simd  # noqa: E402
from finn_milp import (  # noqa: E402
    INPUT_HW, VARIANT_RTL_DSP_NOACT1, XCZU7EV, _calibration_force_dsp, _variant_cost_kwargs,
    candidate_folds, load_config, trace_layer_geometry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

# (slow_layer, fast_layer) -- decoder_type="nearest_conv_upsample" only.
# See module docstring: skip_resize_conv.0 (skip/main branch's own resize-
# conv) is the consistently slow side, expand.0 (residual branch's own
# last op) the fast side, at BOTH upsampling bottlenecks this architecture
# has. This join IS correctly found by layer_topology.compute_predecessor_
# map (confirmed: regular4.0.reduce.0 -> ['up4.skip_resize_conv.0',
# 'up4.expand.0'], regular5.0.reduce.0 -> ['up5.skip_resize_conv.0',
# 'up5.expand.0'] -- unlike a chained RegularBottleneck-to-RegularBottleneck
# residual add, both branches here are IMMEDIATELY tracked ancestors of the
# add, so the one-branch-point cap never bites). Hardcoded anyway rather
# than driven by that map, purely to keep this script self-contained and
# independent of that map's own broader pruning behavior elsewhere in the
# network -- see rebalance_join_folding.py's own module docstring for why a
# full finn_milp.py --max-join-imbalance-ratio re-solve is the more
# principled alternative if you want every real join handled uniformly.
DEFAULT_JOIN_PAIRS = (
    ("up4.skip_resize_conv.0", "up4.expand.0"),
    ("up5.skip_resize_conv.0", "up5.expand.0"),
)


def _layer_total_lut(cost: dict, w: int, a: int, variant: str, force_dsp: bool) -> float:
    variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
    return calibrated_lut(
        cost["total_lut"], w, a, force_dsp=_calibration_force_dsp(variant_kwargs),
        lut_mult=(variant == "hls_lut_noact0"),
    )


def _layer_total_bram(cost: dict, w: int, a: int, variant: str, force_dsp: bool) -> float:
    variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
    return calibrated_bram18k(
        cost["swu_bram18"] + cost["wm_bram18"] + cost.get("thr_bram18", 0), w, a,
        force_dsp=variant_kwargs["force_dsp"],
    )


def rebalance(
    per_layer: dict, geom_by_name: dict, target_ratio: float, cost_metric: str, force_dsp: bool,
    join_pairs: tuple[tuple[str, str], ...],
) -> tuple[dict, list[dict]]:
    """Mutates a COPY of per_layer in place for every slow layer that needed
    a fix, returns (patched_per_layer, report_rows) -- report_rows has one
    entry per join pair actually touched (empty entries for pairs already
    within target_ratio are skipped, not reported)."""
    patched = {name: dict(v) for name, v in per_layer.items()}
    report: list[dict] = []

    for slow_name, fast_name in join_pairs:
        if slow_name not in patched or fast_name not in patched:
            print(f"  [skip] {slow_name} or {fast_name} not in this folding file -- wrong config/decoder_type?")
            continue
        slow, fast = patched[slow_name], patched[fast_name]
        ratio = slow["cycles"] / fast["cycles"] if fast["cycles"] > 0 else float("inf")
        if ratio <= target_ratio:
            print(f"  {slow_name} vs {fast_name}: ratio {ratio:.2f} already <= {target_ratio} -- no change.")
            continue

        layer = geom_by_name[slow_name]
        w, a = slow["weight_bits"], slow["act_bits"]
        target_cycles = target_ratio * fast["cycles"]

        candidates = []
        for pe, simd, ram_style, variant in candidate_folds(layer):
            variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
            cost = layer_cost_pe_simd(
                layer, w, a, pe, simd, RAM_STYLE_BLOCK,
                swu_ram_style="distributed", thr_ram_style=ram_style, **variant_kwargs,
            )
            if cost["cycles"] > target_cycles:
                continue
            lut = _layer_total_lut(cost, w, a, variant, force_dsp)
            bram = _layer_total_bram(cost, w, a, variant, force_dsp)
            metric_value = lut if cost_metric == "lut" else bram if cost_metric == "bram" else cost["total_dsp"]
            candidates.append((metric_value, pe, simd, ram_style, variant, cost, lut, bram))

        if not candidates:
            print(f"  [FAILED] {slow_name}: no candidate fold reaches cycles <= {target_cycles:.0f} "
                  f"(target_ratio={target_ratio} vs {fast_name}'s {fast['cycles']:.0f} cycles) -- left unchanged.")
            continue

        candidates.sort(key=lambda c: c[0])
        _, pe, simd, ram_style, variant, cost, lut, bram = candidates[0]

        old_lut = _layer_total_lut(slow, slow["weight_bits"], slow["act_bits"], slow["variant"], force_dsp)
        old_bram = _layer_total_bram(slow, slow["weight_bits"], slow["act_bits"], slow["variant"], force_dsp)
        old_dsp = slow["total_dsp"]
        old_cycles = slow["cycles"]

        patched[slow_name] = {**slow, "pe": pe, "simd": simd, "thr_ram_style": ram_style, "variant": variant, **cost}
        new_ratio = cost["cycles"] / fast["cycles"]
        print(f"  {slow_name}: pe={slow['pe']}->{pe} simd={slow['simd']}->{simd} "
              f"cycles={old_cycles:.0f}->{cost['cycles']:.0f} (ratio {ratio:.2f}->{new_ratio:.2f}) "
              f"| LUT {old_lut:.0f}->{lut:.0f} (+{lut - old_lut:.0f}) BRAM {old_bram:.0f}->{bram:.0f} "
              f"DSP {old_dsp:.0f}->{cost['total_dsp']:.0f} (+{cost['total_dsp'] - old_dsp:.0f})")
        report.append({
            "slow_layer": slow_name, "fast_layer": fast_name,
            "old_ratio": ratio, "new_ratio": new_ratio,
            "old_cycles": old_cycles, "new_cycles": cost["cycles"],
            "delta_lut": lut - old_lut, "delta_bram": bram - old_bram, "delta_dsp": cost["total_dsp"] - old_dsp,
        })

    return patched, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="MILP/config_*.py module name (no .py).")
    parser.add_argument("--folding-file", type=Path, required=True, help="finn_milp.py --out-file JSON.")
    parser.add_argument("--target-ratio", type=float, default=1.5,
                         help="Max acceptable slow/fast cycle ratio per join pair (default 1.5).")
    parser.add_argument("--cost-metric", default="lut", choices=["lut", "bram", "dsp"],
                         help="Which resource to minimize when picking the slow layer's new fold among every "
                              "candidate that reaches the target ratio (default lut).")
    parser.add_argument("--force-dsp", action="store_true", help="Must match the ORIGINAL finn_milp.py solve's own --force-dsp.")
    parser.add_argument("--out-file", type=Path, required=True)
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

    with open(args.folding_file) as f:
        folding = json.load(f)

    print(f"Rebalancing joins (target_ratio={args.target_ratio}, cost_metric={args.cost_metric}):")
    patched_per_layer, report = rebalance(
        folding["per_layer"], geom_by_name, args.target_ratio, args.cost_metric, args.force_dsp, DEFAULT_JOIN_PAIRS,
    )

    total_lut = sum(
        _layer_total_lut(v, v["weight_bits"], v["act_bits"], v["variant"], args.force_dsp)
        for v in patched_per_layer.values()
    )
    total_bram = sum(
        _layer_total_bram(v, v["weight_bits"], v["act_bits"], v["variant"], args.force_dsp)
        for v in patched_per_layer.values()
    )
    total_dsp = sum(v["total_dsp"] for v in patched_per_layer.values())
    total_cycles = sum(v["cycles"] for v in patched_per_layer.values())

    patched_result = dict(folding)
    patched_result["per_layer"] = patched_per_layer
    patched_result["_diagnostics"] = dict(folding["_diagnostics"])
    patched_result["_diagnostics"].update({
        "total_lut_calibrated": total_lut, "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
        "total_bram18k_calibrated": total_bram, "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
        "total_dsp": total_dsp, "dsp_pct_of_budget": 100 * total_dsp / XCZU7EV["DSP"],
        "total_cycles": total_cycles,
        "join_rebalance": {"target_ratio": args.target_ratio, "cost_metric": args.cost_metric, "report": report},
    })

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(patched_result, f, indent=2)

    print(f"\nBefore -> after (whole-network totals):")
    diag = folding["_diagnostics"]
    print(f"  LUT:    {diag['lut_pct_of_budget']:.1f}% -> {100 * total_lut / XCZU7EV['LUT']:.1f}%")
    print(f"  BRAM:   {diag['bram_pct_of_budget']:.1f}% -> {100 * total_bram / XCZU7EV['BRAM_18K']:.1f}%")
    print(f"  DSP:    {diag['dsp_pct_of_budget']:.1f}% -> {100 * total_dsp / XCZU7EV['DSP']:.1f}%")
    print(f"  cycles: {diag['total_cycles']:.0f} -> {total_cycles:.0f}")
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
