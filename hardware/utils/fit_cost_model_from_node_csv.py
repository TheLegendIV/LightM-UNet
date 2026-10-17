"""Refit finn_cost_model.py's per-node LUT constants from node-resource calibration CSVs -- one reusable entry point for every cost-model recalibration.

Input: one or more `build_node_resource_calibration_csv.py` outputs (a row per real FINN node: its folding / bit widths and the post-route LUT / FF / BRAM / DSP Vivado reports for that
node's hierarchy). Datasets are POOLED, because a constant fitted on one build only knows that build's regime (the INT8 512 build exposed this: thresholds with 255 steps and a wide input
are 5-10x more expensive than the 63-step INT6 nodes the old fits saw).

Method (the repo's convention, see MILP/calibration.csv): per node kind, ordinary least squares THROUGH THE ORIGIN, real_LUT ~ sum_i c_i * x_i, with the x_i structural features of the node
(PE * in_bits, PE * numSteps, channels * numSteps * in_bits, ...). Each fit reports n, per-dataset n, uncentered R^2, the median absolute % error and the sum error of the LIVE formula
(what finn_cost_model.py predicts now) against the refitted one, and a leave-one-dataset-out sum error per dataset (does the fit generalise to a build it has not seen?). Coefficients are
constrained to be >= 0 where `nonneg` is set (a negative LUT per feature is non-physical).

The fit table SPECS below is declarative: adding a node kind or a new feature set is one `Spec(...)`. Specs with `apply=None` are report-only (kinds whose model needs a code change before a fit
can be applied -- e.g. Upsample needs the activation width, which stream_node_cost does not take yet).

Usage (host venv, no Vivado / FINN needed):
    python hardware/utils/fit_cost_model_from_node_csv.py                         # all known datasets, print every fit
    python hardware/utils/fit_cost_model_from_node_csv.py --specs thr_dist_lut_rom,mvau_rtl_derate
    python hardware/utils/fit_cost_model_from_node_csv.py --csv path/to/new_build.csv     # pool one more build in
    python hardware/utils/fit_cost_model_from_node_csv.py --write-calibration-csv         # log the fits to MILP/calibration.csv as 'proposed'
    python hardware/utils/fit_cost_model_from_node_csv.py --apply [--write-calibration-csv]   # patch the constants in MILP/finn_cost_model.py (status 'applied')

Re-running after --apply is safe: the "before" columns are the live constants, so they then measure the refitted model (and a second fit lands on the same coefficients).
"""
import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "MILP", REPO / "hardware" / "utils", REPO / "hardware" / "checks"):
    sys.path.insert(0, str(p))

import finn_cost_model as fcm  # noqa: E402
from compare_model_vs_real_resources_from_dataset import predict_row  # noqa: E402

CALIBRATION_CSV = REPO / "MILP" / "calibration.csv"
COST_MODEL_PY = REPO / "MILP" / "finn_cost_model.py"
B = REPO / "hardware" / "builds"

DATASETS = {
    "int8_512_v2": B / "S12_dense_512_bilinear_int8_v2/results/node_resource_calibration_S12_dense_512_bilinear_int8_v2.csv",
    "bil256_int6": B / "S12_dense_256_u4_bilinear_analytical_v1/results/node_resource_calibration_S12_dense_256_u4_bilinear_analytical_v1_dwfix.csv",
    "tied_dsr104": B / "tied_dsr104/results/finn_deployment_output_tied_dsr104_full_build_fixed/report/node_resource_calibration_tied_dsr104.csv",
    "nn256_v1": B / "12_dense_relu_nearest_conv_upsample_256/results/mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_v1_full.csv",
    "nn256_v2": B / "12_dense_relu_nearest_conv_upsample_256_v2/results/mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_v2_full.csv",
    "nn256_w8_16_v4": B / "12_dense_relu_nearest_conv_upsample_256_w8_16_v4/results/mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_w8_16_v4_full.csv",
    "warmstart150ep": REPO / "hardware/datasets/mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_alpha025_rtl_mvau_noact1_extended.csv",
}


def num(v, default=0.0):
    return float(v) if v not in (None, "") else default


def has(r, key):
    return r.get(key) not in (None, "")


# ----------------------------------------------------------------------------------------------------------------------------------------------- fit table
@dataclass
class Spec:
    key: str
    component: str
    select: Callable[[dict], bool]
    feats: Callable[[dict], list]            # structural features x_i of one row
    before: Callable[[dict], float]          # the LIVE model's LUT for this row
    constants: tuple = ()                    # constant names the coefficients map to (same order as feats); () = report-only
    to_constants: Optional[Callable[[np.ndarray], dict]] = None   # coefficients -> {constant: value} when the map is not 1:1
    nonneg: bool = True
    min_n: int = 5
    notes: str = ""
    formula: str = ""
    datasets: Optional[tuple] = None         # restrict to datasets that carry the needed columns
    live: dict = field(default_factory=dict)


def _thr_trigger(r):
    return num(r.get("depth_trigger_bram"))


def _is_thr(r):
    return r["node_kind"] == "Thresholding" and has(r, "depth_trigger_bram")


def _thr_ctx(r):
    """(PE, channels, numSteps, in_bits, out_bits) of one threshold row."""
    steps = num(r["numSteps"], 255)
    return int(num(r["PE"], 1)) or 1, int(num(r["NumChannels"], 1)) or 1, steps, num(r["act_bits"], 10), math.log2(steps + 1)


def _before_lut(r):
    return predict_row(r)[0]


def _mvau_rtl(r):
    return r["node_kind"] == "MVAU" and "rtl" in r["op_type"].lower()


def _scale_feats(live_name, before):
    """Features of a 'scale' fit: the LIVE prediction with its own derate divided out, so the 1-term coefficient is the new value of that derate."""
    return lambda r: [before(r) / getattr(fcm, live_name)]


def _dwc_before(r):
    return fcm.dwc_cost(int(num(r["inWidth"])), int(num(r["outWidth"])))["total_lut"]


SPECS = [
    Spec("mvau_rtl_lut", "mvau_rtl_lut", _mvau_rtl, lambda r: [1.0, num(r["PE"], 1) or 1, (num(r["PE"], 1) or 1) * (num(r["SIMD"], 1) or 1)], _before_lut,
         constants=("_MVAU_RTL_LUT_BASE", "_MVAU_RTL_LUT_PER_PE", "_MVAU_RTL_LUT_PER_LANE"),
         formula="real_LUT ~= c0 + c1*PE + c2*PE*SIMD   (MVAU_rtl: the DSPs take the multiplies, so the LUT is structural, not bits-driven)",
         notes="Replaces the avg_bits-dependent derate of hardware/utils/fit_rtl_mvu_lut_derate.py, which did not generalise across builds (per-dataset sum errors -19 % .. +72 %)."),
    Spec("thr_block_lut", "threshold_lut", lambda r: _is_thr(r) and _thr_trigger(r) < fcm.THR_DEPTH_TRIGGER_DISTRIBUTED,
         lambda r: [_thr_ctx(r)[0] * _thr_ctx(r)[3] * _thr_ctx(r)[4]], _before_lut, constants=("_THR_BLOCK_LUT_PER_INOUTBIT_PE",),
         formula="real_LUT ~= c * PE * in_bits * out_bits   (ram_style 'block': trigger 1024 / auto)", datasets=("int8_512_v2", "bil256_int6", "tied_dsr104"),
         notes="Datasets that record depth_trigger_bram, so the block and distributed placements can be separated. Replaces PE*9.04*in_bits, which only held at out_bits = 6."),
    Spec("thr_dist_lut", "threshold_lut", lambda r: _is_thr(r) and _thr_trigger(r) >= fcm.THR_DEPTH_TRIGGER_DISTRIBUTED,
         lambda r: [_thr_ctx(r)[0] * _thr_ctx(r)[3] * _thr_ctx(r)[4],
                    _thr_ctx(r)[1] * _thr_ctx(r)[2] * _thr_ctx(r)[3] * (1.0 if _thr_ctx(r)[3] >= fcm._THR_DIST_WIDE_IN_BITS else 0.0)], _before_lut,
         constants=("_THR_DIST_LUT_PER_INOUTBIT_PE", "_THR_DIST_LUT_PER_ROMBIT"),
         formula="real_LUT ~= c1*PE*in_bits*out_bits + c2*channels*numSteps*in_bits*[in_bits >= 13]   (ram_style 'distributed': a wide conv threshold's ROM is in logic)",
         datasets=("int8_512_v2", "bil256_int6", "tied_dsr104"),
         notes="EXPECTED-VALUE fit: real nodes of identical geometry differ 3-4x (compressibility of the threshold ROM), so single nodes can be far off; the design total is what it targets."),
    Spec("vvau_hls_lut", "vvau_lut", lambda r: r["node_kind"] == "VVAU", _scale_feats("_VVAU_HLS_LUT_DERATE", _before_lut), _before_lut,
         constants=("_VVAU_HLS_LUT_DERATE",), min_n=4,
         formula="real_LUT ~= k * live mvu_lut (depthwise)",
         notes="The generic mvu_lut formula over-predicts every real VVAU_hls (the frozen bilinear depthwise conv, 1-3 per build)."),
    Spec("swu_lut_derate", "swu_lut", lambda r: r["node_kind"] == "SWU", _scale_feats("_SWU_LUT_DERATE", _before_lut), _before_lut, constants=("_SWU_LUT_DERATE",),
         formula="real_LUT ~= k * raw SWU formula (300 + buffer LUTRAM)"),
    Spec("maxpool_lut", "maxpool_lut", lambda r: r["node_kind"] == "MaxPool", lambda r: [1.0, num(r["NumChannels"], 1)], _before_lut,
         constants=("_MAXPOOL_FIXED_LUT", "_MAXPOOL_LUT_PER_CHANNEL"), min_n=4, formula="real_LUT ~= b0 + b1 * channels   (StreamingMaxPool_hls)"),
    Spec("addstreams_lut", "addstreams_lut", lambda r: r["node_kind"] == "AddStreams", lambda r: [num(r["PE"], 1)], _before_lut, constants=("_ADDSTREAMS_LUT_PER_PE",),
         formula="real_LUT ~= c * PE   (every real node is PE 1: the per-PE scaling is an unverified assumption)"),
    Spec("dupstreams_lut", "dupstreams_lut", lambda r: r["node_kind"] == "DuplicateStreams", lambda r: [num(r["PE"], 1)], _before_lut, constants=("_DUPSTREAMS_LUT_PER_PE",),
         formula="real_LUT ~= c * PE"),
    Spec("concat_lut", "concat_lut", lambda r: r["node_kind"] == "Concat", lambda r: [1.0], _before_lut, constants=("_CONCAT_LUT",), min_n=3, formula="real_LUT ~= c"),
    Spec("dwc_lut", "dwc_lut", lambda r: r["node_kind"] == "DWC" and has(r, "inWidth") and has(r, "outWidth"),
         lambda r: [num(r["inWidth"]), num(r["outWidth"])], _dwc_before, constants=("_DWC_LUT_PER_IN_BIT", "_DWC_LUT_PER_OUT_BIT"),
         formula="real_LUT ~= c1 * inWidth + c2 * outWidth   (StreamingDataWidthConverter_rtl, stream widths in bits)",
         notes="FINN's own lut_estimation() does not track the real LUT (R2u 0.20); the input stream width does (R2u 0.99)."),
    Spec("labelselect_lut", "labelselect_lut", lambda r: r["node_kind"] == "LabelSelect", lambda r: [num(r["PE"], 1)], lambda r: num(r["PE"], 1) * fcm._LABELSELECT_LUT_PER_PE,
         constants=("_LABELSELECT_LUT_PER_PE",), min_n=1, formula="real_LUT ~= c * PE   (ONE real point)"),
    Spec("channelwise_lut", "bias_lut", lambda r: r["node_kind"] == "ChannelwiseOp", lambda r: [num(r["PE"], 1)], lambda r: num(r["PE"], 1) * fcm._BIAS_LUT_PER_PE,
         constants=("_BIAS_LUT_PER_PE",), min_n=1, formula="real_LUT ~= c * PE   (the final bias add; ONE real point)"),
    Spec("pool_hls_lut", "pool_hls_lut", lambda r: r["node_kind"] == "Pool",
         lambda r: [1.0], lambda r: fcm._POOL_HLS_FIXED_LUT + fcm._POOL_HLS_LUT_PER_PE_BIT * num(r["PE"], 1) * num(r["bits"], 8),
         constants=(), min_n=1, formula="InferPool Pool_hls (ONE real point) -- report-only"),
    Spec("upsample_lut", "upsample_lut", lambda r: r["node_kind"] == "Upsample", lambda r: [1.0, num(r["NumChannels"], 1) * num(r["act_bits"], 8)], _before_lut,
         constants=(), min_n=4, formula="real_LUT ~= b0 + b1 * channels * act_bits -- report-only: stream_node_cost('upsample') takes no bit width yet",
         notes="Real LUT roughly doubles from 6 to 8 bit at the same channel count, so the flat 184 cannot be right across precisions."),
]


# ----------------------------------------------------------------------------------------------------------------------------------------------- fitting
def load(names, extra):
    rows = []
    for n in names:
        p = DATASETS[n]
        if not p.exists():
            print(f"  (skipping {n}: {p} not found)")
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["_ds"] = n
                rows.append(r)
    for p in extra:
        p = Path(p)
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["_ds"] = p.stem
                rows.append(r)
    return rows


def ols_origin(X, y, nonneg):
    """Least squares through the origin; nonneg drops the most negative term and refits (active set), so a non-physical negative LUT/feature never survives."""
    active = list(range(X.shape[1]))
    while active:
        c, *_ = np.linalg.lstsq(X[:, active], y, rcond=None)
        if not nonneg or (c >= 0).all():
            out = np.zeros(X.shape[1])
            out[active] = c
            return out
        active.pop(int(np.argmin(c)))
    return np.zeros(X.shape[1])


def stats(pred, y):
    sse = float(((y - pred) ** 2).sum())
    r2 = 1 - sse / float((y ** 2).sum()) if (y ** 2).sum() else float("nan")
    mape = float(np.median(np.abs(pred - y) / np.maximum(y, 1.0))) * 100
    return r2, mape, (pred.sum() / y.sum() - 1) * 100 if y.sum() else float("nan")


def fit_spec(spec, rows):
    sel = [r for r in rows if spec.select(r) and (spec.datasets is None or r["_ds"] in spec.datasets)]
    if len(sel) < spec.min_n:
        return None, sel
    X = np.array([spec.feats(r) for r in sel], dtype=float)
    y = np.array([num(r["real_LUT"]) for r in sel])
    before = np.array([spec.before(r) for r in sel])
    coef = ols_origin(X, y, spec.nonneg)
    after = X @ coef
    ds = sorted(set(r["_ds"] for r in sel))
    lodo = {}
    if len(ds) > 1:
        for d in ds:
            tr = np.array([r["_ds"] != d for r in sel])
            c = ols_origin(X[tr], y[tr], spec.nonneg)
            te = ~tr
            lodo[d] = ((X[te] @ c).sum() / y[te].sum() - 1) * 100 if y[te].sum() else float("nan")
    return dict(spec=spec, n=len(sel), coef=coef, y=y, before=before, after=after, datasets=ds, rows=sel, lodo=lodo,
                per_ds={d: sum(1 for r in sel if r["_ds"] == d) for d in ds}), sel


def new_constants(res):
    spec, coef = res["spec"], res["coef"]
    if spec.to_constants:
        return spec.to_constants(coef)
    return {name: float(v) for name, v in zip(spec.constants, coef)} if spec.constants else {}


def live_value(name):
    return getattr(fcm, name, None)


def fmt(v):
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:.6g}" for x in v) + ")"
    return f"{v:.6g}"


def report(res):
    spec = res["spec"]
    r2b, mb, sb = stats(res["before"], res["y"])
    r2a, ma, sa = stats(res["after"], res["y"])
    print(f"\n[{spec.key}] {spec.formula}")
    print(f"  n={res['n']}  per dataset: {res['per_ds']}")
    print(f"  live   : medAPE {mb:6.1f}%   sum err {sb:+7.1f}%   (R2u {r2b:.3f})")
    print(f"  refit  : medAPE {ma:6.1f}%   sum err {sa:+7.1f}%   (R2u {r2a:.3f})   coef {np.round(res['coef'], 5).tolist()}")
    if res["lodo"]:
        print("  leave-one-dataset-out sum err: " + "  ".join(f"{d} {v:+.1f}%" for d, v in res["lodo"].items()))
    for name, val in new_constants(res).items():
        old = live_value(name)
        print(f"    {name}: {fmt(old) if old is not None else '(new)'} -> {fmt(val)}")
    if spec.notes:
        print(f"  note: {spec.notes}")


def patch_constants(consts):
    """Rewrite single-line `NAME = value` assignments in finn_cost_model.py (the trailing comment is kept). A constant that does not exist there is an error: add it by hand first."""
    text = COST_MODEL_PY.read_text()
    for name, val in consts.items():
        pat = re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]*?)(\s*(?:#.*)?)$", re.M)
        if not pat.search(text):
            raise SystemExit(f"constant {name} not found as a single-line assignment in {COST_MODEL_PY.name}")
        text = pat.sub(lambda m: m.group(1) + fmt(val) + m.group(3), text, count=1)
    with open(COST_MODEL_PY, "w", newline="\n") as f:      # keep the repo's LF endings on Windows
        f.write(text)


def write_calibration_rows(results, status, dataset_label):
    new = not CALIBRATION_CSV.exists()
    with open(CALIBRATION_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, lineterminator="\n", fieldnames=["date", "component", "constant", "old_value", "new_value", "method", "dataset", "n", "r2_uncentered", "median_abs_pct_err", "status", "notes"])
        if new:
            w.writeheader()
        for res in results:
            spec = res["spec"]
            r2, mape, _ = stats(res["after"], res["y"])
            for name, val in new_constants(res).items():
                old = live_value(name)
                w.writerow({"date": date.today().isoformat(), "component": spec.component, "constant": name, "old_value": fmt(old) if old is not None else "",
                            "new_value": fmt(val), "method": f"OLS through origin (hardware/utils/fit_cost_model_from_node_csv.py {spec.key}): {spec.formula}",
                            "dataset": f"{dataset_label} ({', '.join(f'{d}:{n}' for d, n in res['per_ds'].items())})", "n": res["n"], "r2_uncentered": round(r2, 3),
                            "median_abs_pct_err": f"{mape:.1f}%", "status": status, "notes": spec.notes})


def totals_table(rows, results):
    """Per dataset: modelled-LUT total before / after the refit (rows with a fit use it, every other modelled row keeps the live prediction) vs the real total."""
    fitted = {}
    for res in results:
        for r, a in zip(res["rows"], res["after"]):
            fitted[id(r)] = a
    modelled = {"MVAU", "VVAU", "SWU", "Thresholding", "AddStreams", "DuplicateStreams", "Concat", "Upsample", "MaxPool"}
    print("\n=== modelled-kind LUT totals per dataset (live model vs refit vs real) ===")
    print(f"  {'dataset':16s}{'rows':>6s}{'live':>10s}{'refit':>10s}{'real':>10s}{'live err':>10s}{'refit err':>10s}")
    for d in sorted(set(r["_ds"] for r in rows)):
        live = refit = real = n = 0.0
        for r in rows:
            if r["_ds"] != d or r["node_kind"] not in modelled:
                continue
            try:
                b = predict_row(r)[0]
            except Exception:
                continue
            live += b
            refit += fitted.get(id(r), b)
            real += num(r["real_LUT"])
            n += 1
        if real:
            print(f"  {d:16s}{int(n):6d}{live:10.0f}{refit:10.0f}{real:10.0f}{(live / real - 1) * 100:+9.1f}%{(refit / real - 1) * 100:+9.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS), help=f"comma-separated subset of: {', '.join(DATASETS)} (default all)")
    ap.add_argument("--csv", action="append", default=[], help="an extra node-resource CSV to pool in (repeatable)")
    ap.add_argument("--specs", default="", help="comma-separated spec keys to run (default all)")
    ap.add_argument("--write-calibration-csv", action="store_true", help="append the fits to MILP/calibration.csv (status 'proposed', or 'applied' with --apply)")
    ap.add_argument("--apply", action="store_true", help="patch the fitted constants into MILP/finn_cost_model.py")
    ap.add_argument("--label", default="", help="dataset label written to calibration.csv (default: the pooled dataset names)")
    a = ap.parse_args()

    names = [n for n in a.datasets.split(",") if n]
    rows = load(names, a.csv)
    keys = {k for k in a.specs.split(",") if k}
    print(f"pooled {len(rows)} node rows from {sorted(set(r['_ds'] for r in rows))}")
    results = []
    for spec in SPECS:
        if keys and spec.key not in keys:
            continue
        res, sel = fit_spec(spec, rows)
        if res is None:
            print(f"\n[{spec.key}] skipped: {len(sel)} rows < min_n {spec.min_n}")
            continue
        report(res)
        results.append(res)
    totals_table(rows, results)

    applicable = [r for r in results if r["spec"].constants or r["spec"].to_constants]
    label = a.label or "+".join(sorted(set(r["_ds"] for r in rows)))
    if a.apply:
        consts = {}
        for res in applicable:
            consts.update(new_constants(res))
        patch_constants(consts)
        print(f"\npatched {len(consts)} constants in {COST_MODEL_PY.name}: {', '.join(consts)}")
    if a.write_calibration_csv:
        write_calibration_rows(applicable, "applied" if a.apply else "proposed", label)
        print(f"logged {len(applicable)} fits to {CALIBRATION_CSV.name}")


if __name__ == "__main__":
    main()
