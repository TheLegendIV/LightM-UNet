"""Fits LUT/BRAM_18K derating factors for the FORCED-DSP synthesis regime, at
per-PARTITION granularity, from the two real S12 8-way-partitioned OOC builds
in hardware/results.csv -- see the plan this implements
(C:\\Users\\win32\\.claude\\plans\\nested-singing-flurry.md) for the full
rationale. Summary:

- The two EXISTING anchors in finn_cost_model.py's own calibrated_lut/
  calibrated_bram18k (avg_bits=3.52/8) were both calibrated from real S19
  builds that used FINN's DEFAULT/AUTO resType -- confirmed from those rows'
  own results.csv notes. Both new S12 builds analyzed here FORCED DSP on
  every MVAU/VVAU node instead -- a different regime, so this fits a
  SEPARATE table rather than blending into the existing one.
- "dummy" build (hardcap100) actually used the NODE-LEVEL folding ILP output
  (folding_nodewise_12_separable_dense_relu_min4_hardcap100_maxspeed.json);
  "trained" (hardcap131) used the plain BLOCK-LEVEL folding ILP output
  (artifacts/folding_block_12_separable_dense_relu_min4_hardcap131_maxspeed.json)
  -- the reverse of how they were first described, but immaterial to this
  fit since S12 has 0 depthwise/VVAU layers (node-level folding only changes
  anything for depthwise layers).
- True per-LAYER real (post-synthesis) resource ground truth does not exist
  anywhere in this repo for any build -- only whole-partition aggregate
  totals. This fits at that granularity instead: 8 real partitions x 2
  builds = 16 real (LUT, BRAM_18K) totals, each partition a real,
  contiguous slice of the network with its own mix of per-layer bits/PE/SIMD
  (reconstructed from the real per-layer ILP JSON, not a single build-wide
  average) -- the finest granularity actually supported by the real data.

KNOWN, BOUNDED GAP: the FINN-exported graph contains 4 real weight-bearing
nodes the per-layer ILP never priced at all -- down1.shortcut_proj.0,
down2.shortcut_proj.0, up4.main_up, up5.main_up (FINN-export-only additions:
the Python training model handles down1/down2's channel-count-changing
residual via implicit zero-padding, and up4/up5's "main" branch via a plain
1x1 projection + bilinear/unpool, but the FINN-safe hardware graph instead
uses an explicit learned 1x1 conv / ConvTranspose2d for each, respectively --
see hardware/finn_ooc_*_8way_full.py's own derive_fallback_pe_simd, which
gives these 4 nodes a REAL (PE, SIMD) via a borrowing heuristic but never a
weight_bits/act_bits, since the HAWQ search never saw them as distinct
layers to begin with). These 4 nodes are EXCLUDED from this script's own
raw_lut/raw_bram18 sums (no reliable way to price them without full ONNX
graph geometry access, which this environment doesn't have) -- any
partition containing one is flagged in the output; excluding a real,
partition-local contributor makes that partition's own empirical factor a
slight overestimate (real/raw is measuring less "raw" than the true build
actually has), a real, bounded, explicitly-flagged approximation, not a
silent one.

Also note (unrelated to the fit itself, worth carrying in the output so
nobody later misreads it): the trained build's own per-partition DSP column
in results.csv reads exactly 0 everywhere, which that build's own row notes
attribute to a known ooc_synth_and_timing.json parser bug -- NOT evidence
DSP forcing failed for that build.

Usage:
    python compression/hawq/fit_forced_dsp_derating.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_CSV = REPO_ROOT / "hardware" / "results.csv"
CONV_ORDER_FILE = (
    REPO_ROOT / "hardware" / "outputs" / "finn_exports"
    / "quantEnet_12_separable_dense_relu_min4_hawq_dummy_int8_conv_order.json"
)
# Reused for BOTH builds -- same S12 architecture/topology, confirmed by this
# script's own Step 1 check that both builds' real per-partition node COUNTS
# match exactly, which could only happen if both used identical boundaries.
OUT_FILE = REPO_ROOT / "compression" / "hawq" / "artifacts" / "forced_dsp_derating_fit.json"

BUILDS = {
    "dummy": {
        "model_name": "quantEnet_12_separable_dense_relu_min4_hawq_dummy",
        "config_prefix": "hardcap100",
        "per_layer_file": REPO_ROOT / "compression" / "hawq" / "folding_nodewise_12_separable_dense_relu_min4_hardcap100_maxspeed.json",
        "bridged_config_dir": REPO_ROOT / "hardware" / "outputs" / "hawq_12_sep_dense_relu_min4_dummy_8way_full_20260903_001821",
    },
    "trained": {
        "model_name": "quantEnet_12_separable_dense_relu_min4_hawq_trained",
        "config_prefix": "hardcap131",
        "per_layer_file": REPO_ROOT / "compression" / "hawq" / "artifacts" / "folding_block_12_separable_dense_relu_min4_hardcap131_maxspeed.json",
        "bridged_config_dir": REPO_ROOT / "hardware" / "outputs" / "hawq_12_sep_dense_relu_min4_trained_hardcap131_8way_full_20260904_061905",
    },
}


def load_real_partition_totals(build_key: str) -> dict[int, dict[str, float]]:
    """{partition_id: {"LUT": ..., "BRAM_18K": ...}} from hardware/results.csv,
    matched by model_name + a "_partition_<N>_ooc_synth" config suffix."""
    model_name = BUILDS[build_key]["model_name"]
    totals: dict[int, dict[str, float]] = {}
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_name"] != model_name:
                continue
            m = re.search(r"_partition_(\d+)_ooc_synth$", row["config"])
            if not m:
                continue
            partition_id = int(m.group(1))
            totals[partition_id] = {"LUT": float(row["LUT"]), "BRAM_18K": float(row["BRAM_18K"])}
    if len(totals) != 8:
        raise ValueError(f"{build_key}: expected 8 real partition rows in {RESULTS_CSV}, found {len(totals)}.")
    return totals


def load_partition_weight_node_counts(build_key: str) -> list[int]:
    """Real per-partition weight-bearing node COUNTS, from the real bridged
    folding config files (hawq_folding_config_partition0..7.json) -- these
    are what the ACTUAL build used, so their own node count per partition is
    ground truth for reconstructing partition membership (see module
    docstring: no separate boundary-index file was preserved, but this
    count-based reconstruction needs nothing else)."""
    counts = []
    for i in range(8):
        cfg = json.loads((BUILDS[build_key]["bridged_config_dir"] / f"hawq_folding_config_partition{i}.json").read_text())
        counts.append(len(cfg) - (1 if "Defaults" in cfg else 0))
    return counts


def load_conv_order() -> list[dict]:
    return json.loads(CONV_ORDER_FILE.read_text())


def partition_logical_names(conv_order: list[dict], weight_node_counts: list[int]) -> list[list[str]]:
    """Slices conv_order's own weight-bearing (non-MaxPool2d) logical names,
    in their real topological order, into 8 partitions using each
    partition's own real weight-node COUNT (see load_partition_weight_node_counts) --
    equivalent to slicing by the real node-index boundaries, without needing
    the literal boundary integers (which require live FINN graph access this
    environment doesn't have)."""
    ordered_weight_names = [e["logical_name"] for e in conv_order if e["module_type"] != "MaxPool2d"]
    if sum(weight_node_counts) != len(ordered_weight_names):
        raise ValueError(
            f"Partition weight-node counts sum to {sum(weight_node_counts)}, but conv_order has "
            f"{len(ordered_weight_names)} real weight-bearing entries -- cannot reconcile."
        )
    partitions = []
    cursor = 0
    for n in weight_node_counts:
        partitions.append(ordered_weight_names[cursor : cursor + n])
        cursor += n
    return partitions


def resolve_ilp_entry(logical_name: str, per_layer: dict) -> dict | None:
    """Exact match first, then the SAME ".conv.0" -> ".conv" fallback
    hardware/finn_hawq_folding_bridge.py's own resolve_folding_entry already
    documents (non-dilated RegularBottleneck blocks' bare nn.Conv2d gets
    wrapped in nn.Sequential(conv, bn, act) in the real graph -- named
    "<name>.conv.0" there -- but the ILP's own logical model never wraps a
    lone conv in a Sequential, so it's keyed bare "<name>.conv"). Returns
    None for the 4 real FINN-export-only additions the ILP never priced at
    all (down1/down2.shortcut_proj.0, up4/up5.main_up) -- see module
    docstring's KNOWN, BOUNDED GAP."""
    if logical_name in per_layer:
        return per_layer[logical_name]
    if logical_name.endswith(".conv.0"):
        stripped = logical_name[: -len(".0")]
        if stripped in per_layer:
            return per_layer[stripped]
    return None


def compute_partition_raw_costs(logical_names: list[str], per_layer: dict) -> dict:
    raw_lut = 0.0
    raw_bram18 = 0.0
    lut_weighted_bits_sum = 0.0
    unmatched = []
    for name in logical_names:
        entry = resolve_ilp_entry(name, per_layer)
        if entry is None:
            unmatched.append(name)
            continue
        raw_lut += entry["total_lut"]
        raw_bram18 += entry["swu_bram18"] + entry["wm_bram18"]
        avg_bits = (entry["weight_bits"] + entry["act_bits"]) / 2
        lut_weighted_bits_sum += entry["total_lut"] * avg_bits
    avg_bits = lut_weighted_bits_sum / raw_lut if raw_lut > 0 else float("nan")
    return {
        "raw_lut": raw_lut, "raw_bram18": raw_bram18, "avg_bits": avg_bits,
        "n_layers_matched": len(logical_names) - len(unmatched), "unmatched": unmatched,
    }


def ols_linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Plain ordinary-least-squares slope/intercept for y = a*x + b, via the
    closed-form normal equations (no numpy dependency needed for 16 points).
    Returns (a, b, r_squared)."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    a = cov_xy / var_x
    b = mean_y - a * mean_x
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r_squared


def main() -> None:
    conv_order = load_conv_order()

    all_points = []  # (build, partition_id, avg_bits, lut_factor, bram_factor, flagged)
    for build_key in BUILDS:
        per_layer = json.loads(BUILDS[build_key]["per_layer_file"].read_text())["per_layer"]
        weight_node_counts = load_partition_weight_node_counts(build_key)
        real_totals = load_real_partition_totals(build_key)
        partitions = partition_logical_names(conv_order, weight_node_counts)

        print(f"\n=== {build_key} ({BUILDS[build_key]['config_prefix']}) ===")
        print(f"Per-partition weight-node counts: {weight_node_counts} (sum={sum(weight_node_counts)})")

        raw_lut_sum, raw_bram_sum = 0.0, 0.0
        real_lut_sum, real_bram_sum = 0.0, 0.0
        for pid, names in enumerate(partitions):
            cost = compute_partition_raw_costs(names, per_layer)
            real = real_totals[pid]
            lut_factor = real["LUT"] / cost["raw_lut"]
            bram_factor = real["BRAM_18K"] / cost["raw_bram18"] if cost["raw_bram18"] > 0 else float("nan")
            flagged = bool(cost["unmatched"])
            all_points.append({
                "build": build_key, "partition_id": pid, "avg_bits": cost["avg_bits"],
                "lut_factor": lut_factor, "bram_factor": bram_factor,
                "raw_lut": cost["raw_lut"], "real_lut": real["LUT"],
                "raw_bram18": cost["raw_bram18"], "real_bram18": real["BRAM_18K"],
                "n_layers": len(names), "unmatched_layers": cost["unmatched"],
            })
            raw_lut_sum += cost["raw_lut"]
            raw_bram_sum += cost["raw_bram18"]
            real_lut_sum += real["LUT"]
            real_bram_sum += real["BRAM_18K"]
            flag_str = f"  <-- EXCLUDES {cost['unmatched']}" if flagged else ""
            print(f"  partition {pid}: n_layers={len(names):2d} avg_bits={cost['avg_bits']:.3f} "
                  f"raw_lut={cost['raw_lut']:9.1f} real_lut={real['LUT']:8.0f} lut_factor={lut_factor:6.3f}  "
                  f"raw_bram18={cost['raw_bram18']:7.2f} real_bram18={real['BRAM_18K']:5.0f} "
                  f"bram_factor={bram_factor:7.3f}{flag_str}")

        print(f"  TOTAL raw_lut={raw_lut_sum:.1f} raw_bram18={raw_bram_sum:.2f} "
              f"(real TOTAL LUT/BRAM_18K not re-checked here -- see hardware/results.csv's own "
              f"..._TOTAL row for that cross-check)")

    xs = [p["avg_bits"] for p in all_points]
    lut_ys = [p["lut_factor"] for p in all_points]
    bram_ys = [p["bram_factor"] for p in all_points]

    lut_a, lut_b, lut_r2 = ols_linear_fit(xs, lut_ys)
    bram_a, bram_b, bram_r2 = ols_linear_fit(xs, bram_ys)
    bits_min, bits_max = min(xs), max(xs)

    print(f"\n=== Pooled OLS fit across all {len(all_points)} real partition points ===")
    print(f"avg_bits range observed: [{bits_min:.3f}, {bits_max:.3f}]")
    print(f"LUT factor  = {lut_a:.4f} * avg_bits + {lut_b:.4f}   (R^2={lut_r2:.3f})")
    print(f"  at bits_min: {lut_a*bits_min+lut_b:.3f}x   at bits_max: {lut_a*bits_max+lut_b:.3f}x")
    print(f"BRAM factor = {bram_a:.4f} * avg_bits + {bram_b:.4f}   (R^2={bram_r2:.3f})")
    print(f"  at bits_min: {bram_a*bits_min+bram_b:.3f}x   at bits_max: {bram_a*bits_max+bram_b:.3f}x")

    n_flagged = sum(1 for p in all_points if p["unmatched_layers"])
    print(f"\n{n_flagged}/{len(all_points)} partitions had >=1 unpriced real node excluded from their own raw sum "
          f"(see each row's own unmatched_layers above) -- their own factor is a slight overestimate.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "points": all_points,
        "lut_fit": {"a": lut_a, "b": lut_b, "r_squared": lut_r2, "bits_min": bits_min, "bits_max": bits_max},
        "bram_fit": {"a": bram_a, "b": bram_b, "r_squared": bram_r2, "bits_min": bits_min, "bits_max": bits_max},
    }, indent=2))
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
