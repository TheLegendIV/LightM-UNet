"""Fits LUT/BRAM_18K derating factors for the FORCED-DSP synthesis regime, at
per-PARTITION granularity, from the one real S12-DENSE 8-way-partitioned OOC
build in hardware/results.csv (quantEnet_12_dense_relu_warmstart150ep_
alpha025_finn_calibrated_int8). Sibling of fit_forced_dsp_derating.py (the
S12-SEPARABLE fit) -- same method, applied to the dense (non-factored,
SEPARABLE_DILATED=False) geometry, whose real-vs-predicted LUT gap (336% of
budget vs a 69.9%-of-budget prediction) is what prompted this fit. See that
script's own docstring for the full method rationale; differences from it:

- Only ONE real build exists for dense (not two) -- 8 partition points total,
  not 16. Weaker statistically; this script reports R^2 honestly and a flat
  (no-bits-slope) fallback fit alongside the affine one so a downstream
  consumer isn't stuck with an overfit 2-point-per-degree-of-freedom line.
- per_layer_file is this session's FRESHLY REGENERATED alpha=0.25 per-layer
  ILP output (candidate_bits {4,6,8}, --hard-lut-fraction 0.7 --force-dsp),
  computed under the CURRENT (post mvu_lut-fix) finn_cost_model.py -- i.e.
  raw_lut here already reflects the new addertree_luts/acc_luts terms, not
  the old flat formula the ACTUAL synthesized hardware was folded under. This
  is intentional: the fit's job is "how wrong is TODAY's formula", not "how
  wrong was the formula that happened to produce this hardware".
- CONV_ORDER_FILE reused from the "dummy" export (confirmed identical op-type
  histogram to the finn_calibrated export via direct onnx.load() comparison
  -- 546 nodes, same Counter, same 3 ConvTranspose2d at up4.up.0/up5.up.0/
  final -- so no topology-vintage mismatch despite this build's own
  results.csv notes mentioning an unrelated main_up change elsewhere).
- KNOWN, BOUNDED GAP (same shape as the separable fit's own): conv_order's 89
  weight-bearing nodes vs the per-layer ILP's 88 -- diff is exactly
  {down1.shortcut_proj, down2.shortcut_proj, up4.main_up.1, up5.main_up.1},
  the same 4-node FINN-export-only category documented in
  fit_forced_dsp_derating.py. Excluded from raw sums, partition flagged.

Usage:
    python compression/hawq/fit_forced_dsp_derating_s12_dense.py
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
    / "quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8_conv_order.json"
)
OUT_FILE = REPO_ROOT / "compression" / "hawq" / "artifacts" / "forced_dsp_derating_fit_s12_dense.json"

BUILDS = {
    "finn_calibrated": {
        "model_name": "quantEnet_12_dense_relu_warmstart150ep_alpha025_finn_calibrated_int8",
        "config_prefix": "alpha025",
        "per_layer_file": REPO_ROOT / "compression" / "hawq" / "artifacts"
            / "12_dense_relu_warmstart150ep_ILP_outputs_perlayer_forcedsp_lut70"
            / "layer_bits_folding_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut70.json",
        "bridged_config_dir": REPO_ROOT / "hardware" / "outputs"
            / "12_dense_relu_warmstart150ep_alpha025_finn_calibrated_8way_full_20260914_221145",
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
    ground truth for reconstructing partition membership."""
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
    partition's own real weight-node COUNT."""
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
    hardware/finn_hawq_folding_bridge.py's own resolve_folding_entry
    documents. Returns None for the real FINN-export-only additions the ILP
    never priced at all (down1/down2.shortcut_proj, up4/up5.main_up.1 for
    this dense architecture)."""
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
    """Plain OLS slope/intercept for y = a*x + b via closed-form normal
    equations. Returns (a, b, r_squared)."""
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


def flat_fit(ys: list[float]) -> tuple[float, float]:
    """RMSE-minimizing constant factor (= mean) and its own residual RMSE,
    for comparison against the affine fit's R^2 (with only 8 points, a
    2-parameter affine fit can look good by construction -- report both)."""
    mean_y = sum(ys) / len(ys)
    rmse = (sum((y - mean_y) ** 2 for y in ys) / len(ys)) ** 0.5
    return mean_y, rmse


def main() -> None:
    conv_order = load_conv_order()

    all_points = []
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
            lut_factor = real["LUT"] / cost["raw_lut"] if cost["raw_lut"] > 0 else float("nan")
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

        print(f"  TOTAL raw_lut={raw_lut_sum:.1f} real_lut={real_lut_sum:.1f} "
              f"(pooled factor={real_lut_sum/raw_lut_sum:.3f}x)  "
              f"raw_bram18={raw_bram_sum:.2f} real_bram18={real_bram_sum:.1f} "
              f"(pooled factor={real_bram_sum/raw_bram_sum:.3f}x)")

    xs = [p["avg_bits"] for p in all_points]
    lut_ys = [p["lut_factor"] for p in all_points]
    bram_ys = [p["bram_factor"] for p in all_points]

    lut_a, lut_b, lut_r2 = ols_linear_fit(xs, lut_ys)
    bram_a, bram_b, bram_r2 = ols_linear_fit(xs, bram_ys)
    lut_flat, lut_flat_rmse = flat_fit(lut_ys)
    bram_flat, bram_flat_rmse = flat_fit(bram_ys)
    bits_min, bits_max = min(xs), max(xs)

    print(f"\n=== Fits across all {len(all_points)} real partition points (1 build, dense S12) ===")
    print(f"avg_bits range observed: [{bits_min:.3f}, {bits_max:.3f}]")
    print(f"LUT affine  = {lut_a:.4f} * avg_bits + {lut_b:.4f}   (R^2={lut_r2:.3f})")
    print(f"  at bits_min: {lut_a*bits_min+lut_b:.3f}x   at bits_max: {lut_a*bits_max+lut_b:.3f}x")
    print(f"LUT flat    = {lut_flat:.4f}x   (RMSE={lut_flat_rmse:.4f})")
    print(f"BRAM affine = {bram_a:.4f} * avg_bits + {bram_b:.4f}   (R^2={bram_r2:.3f})")
    print(f"  at bits_min: {bram_a*bits_min+bram_b:.3f}x   at bits_max: {bram_a*bits_max+bram_b:.3f}x")
    print(f"BRAM flat   = {bram_flat:.4f}x   (RMSE={bram_flat_rmse:.4f})")

    n_flagged = sum(1 for p in all_points if p["unmatched_layers"])
    print(f"\n{n_flagged}/{len(all_points)} partitions had >=1 unpriced real node excluded from their own raw sum.")
    print("NOTE: only 8 points from ONE build (separable's fit had 16 from two) -- affine R^2 is "
          "not strong evidence of a real bits-dependent trend at this sample size; flat factor "
          "shown alongside for comparison.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "points": all_points,
        "lut_fit": {"a": lut_a, "b": lut_b, "r_squared": lut_r2, "bits_min": bits_min, "bits_max": bits_max},
        "bram_fit": {"a": bram_a, "b": bram_b, "r_squared": bram_r2, "bits_min": bits_min, "bits_max": bits_max},
        "lut_flat_fit": {"factor": lut_flat, "rmse": lut_flat_rmse},
        "bram_flat_fit": {"factor": bram_flat, "rmse": bram_flat_rmse},
    }, indent=2))
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
