"""Reusable model-vs-real resource cross-check: any finn_milp.py
`layer_bits_folding_*.json` against any `build_node_resource_calibration_csv.py`
per-node CSV for the SAME solved build.

Buckets both sides into the coarsest groupings that stay honest under how
finn_milp.py actually applies `calibrated_lut`/`calibrated_bram18k`:

  - "conv_stack" (real node_kind MVAU + VVAU + SWU + Thresholding): model side
    is EVERY per_layer entry's already-BUNDLED `lut_calibrated`/
    `bram18k_calibrated`/`dsp`/`uram18` (calibration is applied once to the
    swu+mvu+thr+mp sum per node -- see finn_milp.py's `raw_lut[key] =
    calibrated_lut(cost["total_lut"], ...)` -- so this script never tries to
    de-bundle that calibrated figure back into separate SWU/MVAU/Thresholding
    numbers, which would silently misattribute the calibration factor), PLUS
    every extra_nodes entry whose kind is a standalone Thresholding_rtl node
    not fused to a per_layer conv (input_quant/pool_quant/act/out_act/
    residual_add/skip_quant) or an extra MVAU-shaped node (pad_mvau) -- these
    all land in the same real op_type pool as the per_layer ones, with no way
    to tell them apart in the CSV, so they're combined here too.
  - "AddStreams"/"DuplicateStreams"/"Concat"/"Upsample"/"MaxPool": each is
    exactly one extra_nodes kind (add/dup/concat/upsample/maxpool), and each
    extra_nodes entry is calibrated individually (never bundled with
    anything else), so these compare 1:1 against their own real node_kind
    with no attribution risk.
  - "FIFO" / "DWC": real-only reference rows -- finn_milp.py does not model
    FIFOs OR StreamingDataWidthConverter nodes at all (DWCs aren't even in
    its dataflow-graph vocabulary; FINN inserts them automatically on
    stream-width mismatches between adjacent folded nodes), so there is
    nothing on the model side to compare either against.

Usage:
    python compare_model_vs_real_resources.py \\
        --folding-json MILP/artifacts/S12_dense_nn_upsample_256_v2/layer_bits_folding_..._dsrate1.5.json \\
        --calibration-csv hardware/builds/12_dense_relu_nearest_conv_upsample_256_v2/results/mvau_lut_calibration_dataset_..._full.csv

Requires the calibration CSV to have been (re)generated with
build_node_resource_calibration_csv.py AFTER its AddStreams_hls/
DuplicateStreams_hls/StreamingConcat_hls/UpsampleNearestNeighbour_hls/
StreamingMaxPool_hls op-type support was added (2026-09-28) -- an older CSV
predating that fix will just show 0 real rows for those four node kinds
(nothing crashes, but the corresponding rows will read as +inf% error).
"""
import argparse
import csv
import json
from collections import defaultdict

# extra_nodes kinds that are standalone Thresholding_rtl / extra MVAU-shaped
# nodes -- NOT their own real node_kind, they land in the same conv_stack
# pool as per_layer's bundled thr_lut/mvu_lut in the real CSV.
CONV_STACK_EXTRA_KINDS = {
    "input_quant", "pool_quant", "act", "out_act", "residual_add", "skip_quant", "pad_mvau",
}
# extra_nodes kinds that DO map 1:1 onto their own real node_kind.
STANDALONE_EXTRA_KIND_TO_REAL_KIND = {
    "add": "AddStreams",
    "dup": "DuplicateStreams",
    "concat": "Concat",
    "upsample": "Upsample",
    "maxpool": "MaxPool",
}
REAL_KINDS_IN_CONV_STACK = {"MVAU", "VVAU", "SWU", "Thresholding"}

RESOURCE_FIELDS = ("lut", "bram18_equiv", "dsp", "uram")


def new_totals():
    return {f: 0.0 for f in RESOURCE_FIELDS}


def load_model_totals(folding_json_path):
    d = json.load(open(folding_json_path))
    totals = defaultdict(new_totals)

    for layer in d["per_layer"].values():
        b = totals["conv_stack"]
        b["lut"] += layer.get("lut_calibrated", 0.0)
        b["bram18_equiv"] += layer.get("bram18k_calibrated", 0.0)
        b["dsp"] += layer.get("total_dsp", layer.get("mvu_dsp", 0))
        b["uram"] += layer.get("swu_uram18", 0) + layer.get("wm_uram18", 0) + layer.get("thr_uram18", 0)

    for node in d["extra_nodes"].values():
        kind = node["kind"]
        if kind in CONV_STACK_EXTRA_KINDS:
            bucket = "conv_stack"
        elif kind in STANDALONE_EXTRA_KIND_TO_REAL_KIND:
            bucket = STANDALONE_EXTRA_KIND_TO_REAL_KIND[kind]
        else:
            bucket = f"extra:{kind}"  # unrecognized kind -- surfaced, not silently dropped
        b = totals[bucket]
        b["lut"] += node.get("lut_calibrated", 0.0)
        b["bram18_equiv"] += node.get("bram18k_calibrated", 0.0)
        b["dsp"] += node.get("dsp", 0)
        b["uram"] += node.get("uram18", 0)

    return totals, d["_diagnostics"]


def load_real_totals(calibration_csv_path):
    totals = defaultdict(new_totals)
    with open(calibration_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{calibration_csv_path}: no rows -- wrong path, or build/CSV-generation failed?")

    def num(v):
        return float(v) if v not in (None, "") else 0.0

    for r in rows:
        kind = r["node_kind"]
        # MVAU/VVAU/SWU/Thresholding all fold into one real bucket -- see
        # module docstring: the model's per_layer entries bundle+calibrate
        # these together (calibrated_lut/calibrated_bram18k run ONCE on the
        # swu+mvu+thr+mp sum per node), so splitting the real side back out
        # to match individually would compare against a de-bundled number
        # the model never actually produced.
        bucket = "conv_stack" if kind in REAL_KINDS_IN_CONV_STACK else kind
        b = totals[bucket]
        b["lut"] += num(r["real_LUT"])
        b["bram18_equiv"] += num(r["real_BRAM18"]) + 2 * num(r["real_BRAM36"])
        b["dsp"] += num(r["real_DSP"])
        b["uram"] += num(r["real_URAM"])
    return totals, len(rows)


def pct_err(model_v, real_v):
    if real_v == 0:
        return float("nan") if model_v == 0 else float("inf")
    return 100.0 * (model_v - real_v) / real_v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folding-json", required=True, help="finn_milp.py's layer_bits_folding_*.json for this build's solve")
    p.add_argument("--calibration-csv", required=True, help="build_node_resource_calibration_csv.py's per-node CSV for the SAME build")
    args = p.parse_args()

    model_totals, diag = load_model_totals(args.folding_json)
    real_totals, n_real_rows = load_real_totals(args.calibration_csv)

    print(f"=== Real CSV: {n_real_rows} rows ({args.calibration_csv}) ===")
    for kind in ("FIFO", "DWC"):
        unmodeled = real_totals.pop(kind, None)
        if unmodeled is None:
            continue
        print(f"{kind} (unmodeled by finn_milp.py, excluded below): LUT={unmodeled['lut']:.0f}  "
              f"BRAM18eq={unmodeled['bram18_equiv']:.1f}  DSP={unmodeled['dsp']:.0f}  URAM={unmodeled['uram']:.0f}")
    print()

    buckets = sorted(set(model_totals) | set(real_totals))
    header = f"{'bucket':16s} {'LUT model':>10s} {'LUT real':>10s} {'err%':>7s}  " \
             f"{'BRAM model':>10s} {'BRAM real':>10s} {'err%':>7s}  " \
             f"{'DSP model':>9s} {'DSP real':>9s} {'err%':>7s}"
    print(header)
    print("-" * len(header))
    for bucket in buckets:
        m = model_totals.get(bucket, new_totals())
        r = real_totals.get(bucket, new_totals())
        if bucket not in real_totals:
            print(f"{bucket:16s} (model solved this kind, but 0 matching real rows -- "
                  f"CSV predates the AddStreams/DuplicateStreams/Concat/Upsample/MaxPool op-type fix?)")
            continue
        print(f"{bucket:16s} {m['lut']:10.0f} {r['lut']:10.0f} {pct_err(m['lut'], r['lut']):+6.1f}%  "
              f"{m['bram18_equiv']:10.1f} {r['bram18_equiv']:10.1f} {pct_err(m['bram18_equiv'], r['bram18_equiv']):+6.1f}%  "
              f"{m['dsp']:9.0f} {r['dsp']:9.0f} {pct_err(m['dsp'], r['dsp']):+6.1f}%")

    model_total_ex_fifo = new_totals()
    real_total_ex_fifo = new_totals()
    for f in RESOURCE_FIELDS:
        model_total_ex_fifo[f] = sum(b[f] for b in model_totals.values())
        real_total_ex_fifo[f] = sum(b[f] for k, b in real_totals.items())
    print("-" * len(header))
    print(f"{'TOTAL (no FIFO)':16s} "
          f"{model_total_ex_fifo['lut']:10.0f} {real_total_ex_fifo['lut']:10.0f} "
          f"{pct_err(model_total_ex_fifo['lut'], real_total_ex_fifo['lut']):+6.1f}%  "
          f"{model_total_ex_fifo['bram18_equiv']:10.1f} {real_total_ex_fifo['bram18_equiv']:10.1f} "
          f"{pct_err(model_total_ex_fifo['bram18_equiv'], real_total_ex_fifo['bram18_equiv']):+6.1f}%  "
          f"{model_total_ex_fifo['dsp']:9.0f} {real_total_ex_fifo['dsp']:9.0f} "
          f"{pct_err(model_total_ex_fifo['dsp'], real_total_ex_fifo['dsp']):+6.1f}%")


if __name__ == "__main__":
    main()
