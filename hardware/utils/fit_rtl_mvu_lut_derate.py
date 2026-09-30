"""Fit an avg_bits-dependent RTL MVU LUT derate against real per-node data,
and (optionally) log the fit to MILP/calibration.csv.

Why: finn_cost_model.py's RTL LUT derate (applied inside `conv_cost_pe_simd`
whenever `impl_style=="rtl"`) was originally a FLAT scalar
(`_RTL_MVU_LUT_DERATE=0.4868`), fit on ONE dataset (warmstart150ep, avg_bits
mean 5.64) and only ever aggregate-validated on it. A held-out check found
real MVAU LUT error swinging from -9.0% (v2 build, avg_bits mean 4.53) to
+72.4% (v4 build, avg_bits mean 6.70) with identical code -- exactly the
kind of avg_bits-dependence the HLS auto-resType path already accounts for
(see `calibrated_lut()`'s `_LUT_ANCHOR_BITS`/`_LUT_ANCHOR_FACTORS` two-point
interpolation) but the RTL path never had, until this refit
(`_RTL_MVU_LUT_DERATE_BITS`/`_FACTORS`, see finn_cost_model.md "RTL LUT
derating").

We now have 4 real RTL/noActivation=1 datasets whose avg_bits means spread
across [4.53, 7.58] (pooled per-node range ~[3.5, 8.0]) -- enough real
diversity to fit a proper avg_bits-dependent derate instead of a flat one.

Method: pool real MVAU_rtl rows (VVAU_hls excluded -- conv_cost_pe_simd
forces impl_style=HLS for depthwise layers, so this derate never applies to
it; it's a separate, currently-uncalibrated regime) from all 4 datasets,
reconstruct each row's PRE-derate raw mvu_lut the same way
compare_model_vs_real_resources_from_dataset.py's predict_row() does
(synthetic 1x1 LayerGeometry sized so cin*kh*kw/groups==MW, cout==MH -- see
that script's own module docstring for why this is exact for mvu_lut/
wm_bram18/mvu_dsp), then fit:

    real_LUT ~= a*(avg_bits*raw_mvu_lut) + b*raw_mvu_lut   (2-term OLS, through origin)
    => derate(avg_bits) = a*avg_bits + b

matching this repo's existing "N-term OLS through origin" convention (see
MILP/calibration.csv's threshold_lut/threshold_bram rows).

raw_mvu_lut_and_bits() recovers the pre-derate value by dividing the
function's current *post-derate* mvu_lut by WHATEVER derate
finn_cost_model.py currently applies for that node's avg_bits (introspected
live via `_interpolate_derating(avg_bits, _RTL_MVU_LUT_DERATE_FACTORS,
anchor_bits=_RTL_MVU_LUT_DERATE_BITS)` -- the only multiplicative factor
`conv_cost_pe_simd` applies in this path for RTL, so this is exact). This
makes the script safe to rerun after a previous fit from this SAME script
has already been applied to production code -- each rerun divides out
whatever is currently live, not a hardcoded historical constant, so a
future rerun (e.g. against a new v5 build) correctly measures the CURRENT
live formula's error and refits from there.

--write-calibration-csv appends this run's fit as a new dated row to
MILP/calibration.csv via csv.DictWriter (correct quoting/escaping handled
automatically -- no hand-editing, and no risk of the comma-in-unquoted-field
bug a hand-written row can introduce). Without the flag, the script only
prints.

Usage:
    docker exec lightmunet_dev python3 /workspace/LightM-UNet/hardware/fit_rtl_mvu_lut_derate.py [--write-calibration-csv]
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MILP"))
from finn_cost_model import (  # noqa: E402
    LayerGeometry, conv_cost_pe_simd, _interpolate_derating, _RTL_MVU_LUT_DERATE_BITS, _RTL_MVU_LUT_DERATE_FACTORS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CSV = REPO_ROOT / "MILP" / "calibration.csv"

DATASETS = {
    "v1": REPO_ROOT / "hardware/builds/12_dense_relu_nearest_conv_upsample_256/results/"
                       "mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_v1_full.csv",
    "v2": REPO_ROOT / "hardware/builds/12_dense_relu_nearest_conv_upsample_256_v2/results/"
                       "mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_v2_full.csv",
    "v4": REPO_ROOT / "hardware/builds/12_dense_relu_nearest_conv_upsample_256_w8_16_v4/results/"
                       "mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_w8_16_v4_full.csv",
    "warmstart150ep": REPO_ROOT / "hardware/datasets/"
                                  "mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_alpha025_rtl_mvau_noact1_extended.csv",
}


def num(v, default=0.0):
    return float(v) if v not in (None, "") else default


def load_mvau_rtl_rows(path):
    """MVAU_rtl rows only -- excludes VVAU_hls (see module docstring)."""
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["node_kind"] == "MVAU" and "rtl" in r["op_type"].lower()]


def raw_and_current_mvu_lut(row):
    """(raw pre-derate mvu_lut, CURRENT live post-derate mvu_lut, weight_bits, act_bits)."""
    mh, mw = int(num(row["MH"])), int(num(row["MW"]))
    pe, simd = int(num(row["PE"], 1)), int(num(row["SIMD"], 1))
    wbits, abits = num(row["weight_bits"], 8), num(row["act_bits"], 8)
    ram_style = row.get("ram_style") or "auto"
    geom = LayerGeometry(
        op_type="Conv2d", name=row["node_name"], stage="",
        cin=mw, hin=1, win=1, cout=mh, hout=1, wout=1, kh=1, kw=1, sh=1, sw=1, groups=1,
    )
    cost = conv_cost_pe_simd(
        geom, wbits, abits, pe, simd, ram_style=ram_style,
        force_dsp=True, impl_style="rtl", no_activation=False,
    )
    current_pred = cost["mvu_lut"]
    avg_bits = (wbits + abits) / 2
    current_derate = _interpolate_derating(avg_bits, _RTL_MVU_LUT_DERATE_FACTORS, anchor_bits=_RTL_MVU_LUT_DERATE_BITS)
    return current_pred / current_derate, current_pred, wbits, abits


def ols_through_origin_2term(xyz):
    """Closed-form normal-equations solve for y ~= a*x1 + b*x2 through the origin."""
    s11 = s12 = s22 = sy1 = sy2 = 0.0
    for x1, x2, y in xyz:
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        sy1 += x1 * y
        sy2 += x2 * y
    det = s11 * s22 - s12 * s12
    a = (sy1 * s22 - sy2 * s12) / det
    b = (sy2 * s11 - sy1 * s12) / det
    return a, b


def r2_uncentered(xyz, a, b):
    ss_res = ss_tot = 0.0
    for x1, x2, y in xyz:
        ss_res += (y - (a * x1 + b * x2)) ** 2
        ss_tot += y * y
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def median(vals):
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def append_calibration_row(row: dict) -> None:
    with open(CALIBRATION_CSV, newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames
        existing_rows = list(csv.DictReader(f))
    missing = set(fieldnames) - set(row)
    if missing:
        raise ValueError(f"row is missing required calibration.csv column(s): {sorted(missing)}")
    with open(CALIBRATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
    print(f"\nAppended a new row to {CALIBRATION_CSV}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-calibration-csv", action="store_true",
                     help="Append this run's fit to MILP/calibration.csv (default: print only).")
    args = ap.parse_args()

    per_build = {}
    for name, path in DATASETS.items():
        entries = []
        for row in load_mvau_rtl_rows(path):
            raw, current_pred, wbits, abits = raw_and_current_mvu_lut(row)
            entries.append(((wbits + abits) / 2, raw, current_pred, num(row["real_LUT"])))
        per_build[name] = entries
    pooled = [e for entries in per_build.values() for e in entries]

    print(f"{'build':16s} {'n':>4s} {'avg_bits min':>13s} {'avg_bits max':>13s} {'avg_bits mean':>14s} "
          f"{'real/current-live':>18s}")
    for name, entries in per_build.items():
        abits_list = [e[0] for e in entries]
        real_sum = sum(e[3] for e in entries)
        current_pred_sum = sum(e[2] for e in entries)
        ratio = real_sum / current_pred_sum if current_pred_sum else float("nan")
        print(f"{name:16s} {len(entries):4d} {min(abits_list):13.2f} {max(abits_list):13.2f} "
              f"{sum(abits_list) / len(abits_list):14.2f} {ratio:18.3f}")

    bits_min = min(e[0] for e in pooled)
    bits_max = max(e[0] for e in pooled)
    print(f"\nPooled n={len(pooled)}  avg_bits range=[{bits_min:.3f}, {bits_max:.3f}]")

    xyz = [(ab * raw, raw, real) for ab, raw, _, real in pooled]
    a, b = ols_through_origin_2term(xyz)
    r2 = r2_uncentered(xyz, a, b)
    abs_pct_errs = [abs((a * ab + b) * raw - real) / real * 100 for ab, raw, _, real in pooled if real]
    med_err = median(abs_pct_errs)

    derate_lo, derate_hi, derate_6 = a * bits_min + b, a * bits_max + b, a * 6.0 + b
    print("\nFit: real_LUT ~= a*(avg_bits*raw_mvu_lut) + b*raw_mvu_lut  =>  derate(avg_bits) = a*avg_bits + b")
    print(f"  a={a:.6f}  b={b:.6f}")
    print(f"  R^2 (uncentered) = {r2:.4f}   n = {len(pooled)}   median abs %% err (per node) = {med_err:.1f}%%")
    print(f"  derate(avg_bits={bits_min:.3f}) = {derate_lo:.4f}")
    print(f"  derate(avg_bits=6.0)           = {derate_6:.4f}   (test_finn_cost_model.py's W6A6 probe point)")
    print(f"  derate(avg_bits={bits_max:.3f}) = {derate_hi:.4f}")
    print("\nSuggested finn_cost_model.py constants:")
    print(f"  _RTL_MVU_LUT_DERATE_BITS = ({bits_min:.4f}, {bits_max:.4f})")
    print(f"  _RTL_MVU_LUT_DERATE_FACTORS = ({derate_lo:.4f}, {derate_hi:.4f})")

    print("\nPer-build aggregate LUT error, CURRENT LIVE derate vs THIS RUN's freshly refit derate:")
    per_build_err = {}
    for name, entries in per_build.items():
        real_sum = sum(e[3] for e in entries)
        live_pred = sum(e[2] for e in entries)
        new_pred = sum((a * e[0] + b) * e[1] for e in entries)
        live_err = 100 * (live_pred - real_sum) / real_sum
        new_err = 100 * (new_pred - real_sum) / real_sum
        per_build_err[name] = (live_err, new_err)
        print(f"  {name:16s} live_err={live_err:+7.1f}%   new_fit_err={new_err:+7.1f}%")

    if args.write_calibration_csv:
        row = {
            "date": date.today().isoformat(),
            "component": "mvu_lut",
            "constant": "_RTL_MVU_LUT_DERATE_BITS / _RTL_MVU_LUT_DERATE_FACTORS",
            "old_value": f"bits=({_RTL_MVU_LUT_DERATE_BITS[0]:.4f}, {_RTL_MVU_LUT_DERATE_BITS[1]:.4f}), "
                          f"factors=({_RTL_MVU_LUT_DERATE_FACTORS[0]:.4f}, {_RTL_MVU_LUT_DERATE_FACTORS[1]:.4f}) (live before this run)",
            "new_value": f"bits=({bits_min:.4f}, {bits_max:.4f}), factors=({derate_lo:.4f}, {derate_hi:.4f})",
            "method": "2-term OLS through origin via hardware/fit_rtl_mvu_lut_derate.py: "
                      "real_LUT ~= a*(avg_bits*raw_mvu_lut) + b*raw_mvu_lut",
            "dataset": "12_dense_relu_nearest_conv_upsample_256 (v1) + _v2 + _w8_16_v4 + "
                       "warmstart150ep_alpha025_rtl_mvau_noact1_extended (4 datasets pooled, MVAU_rtl rows only)",
            "n": str(len(pooled)),
            "r2_uncentered": f"{r2:.4f}",
            "median_abs_pct_err": f"{med_err:.1f}%",
            "status": "applied",
            "notes": "Per-build error, live formula before this run vs this run's fresh refit: "
                     + "; ".join(f"{name} {live:+.1f}%->{new:+.1f}%" for name, (live, new) in per_build_err.items())
                     + ". Pools every real RTL/noActivation=1 dataset that currently exists -- no held-out set "
                       "remains; the next new real build is the first genuine holdout check.",
        }
        append_calibration_row(row)


if __name__ == "__main__":
    main()
