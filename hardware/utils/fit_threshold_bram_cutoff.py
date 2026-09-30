"""Fit an empirical "real hardware infers zero BRAM" cutoff for standalone
Thresholding_rtl nodes, mirroring the already-existing analogous cutoff for
MVAU/VVAU weight memory (`_WM_BRAM_AUTO_MIN_WMEM`/`_MIN_MEM_WIDTH` in
finn_cost_model.py).

Why: `_thresholding_rtl_cost` currently applies `_THR_RTL_BRAM18_PER_PE_NUMSTEP`
to EVERY threshold node unconditionally. Real data shows roughly half of all
real threshold nodes synthesize with ZERO BRAM and ZERO LUTRAM (Vivado infers
no dedicated memory at all for small-enough thresholds): v2 87/170 nonzero,
v4 78/170 nonzero. Applying a flat per-(PE*numSteps) rate to a population
that's half genuinely zero both under-fits the nonzero rows (the rate gets
diluted, see MILP/calibration.csv's threshold_bram row, R^2=0.753) and
mis-predicts nonzero BRAM for every node that should be free. This is a
likely driver of the BRAM error flipping sign between v2 (-26.7%) and v4
(+47.4%) found this session (different real threshold-size mixes land on
different sides of an unmodeled boundary).

Note real `PE` is degenerate at exactly 1 for every threshold row in both
v2 and v4 -- so a pure PE*numSteps cutoff reduces to numSteps-only, which
this script confirms does NOT cleanly separate the two populations (v2's
numSteps=255 bucket alone contains both zero and nonzero rows). This script
searches NumChannels*numSteps and PE*NumChannels*numSteps too (the latter
being the general form, reducing to the others when PE=1) and reports the
real misclassification count at the best single-threshold split for each --
NOT expected to be a clean zero-counterexample boundary like
_WM_BRAM_AUTO_MIN_WMEM had, reported honestly either way.

--write-calibration-csv appends this run's fit as a new dated row to
MILP/calibration.csv via csv.DictWriter (correct quoting/escaping handled
automatically -- no hand-editing). Without the flag, the script only prints.
The printed/logged "before" comparison is always against finn_cost_model.py's
CURRENT LIVE constants (imported directly, not a hardcoded historical
snapshot), so reruns after a previous fit from this script was applied stay
meaningful.

Usage:
    docker exec lightmunet_dev python3 /workspace/LightM-UNet/hardware/fit_threshold_bram_cutoff.py [--write-calibration-csv]
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MILP"))
from finn_cost_model import _THR_RTL_BRAM18_PER_PE_NUMSTEP as LIVE_RATE  # noqa: E402
from finn_cost_model import _THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP as LIVE_CUTOFF  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CSV = REPO_ROOT / "MILP" / "calibration.csv"

DATASETS = {
    "v2": REPO_ROOT / "hardware/builds/12_dense_relu_nearest_conv_upsample_256_v2/results/"
                       "mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_v2_full.csv",
    "v4": REPO_ROOT / "hardware/builds/12_dense_relu_nearest_conv_upsample_256_w8_16_v4/results/"
                       "mvau_lut_calibration_dataset_12_dense_relu_nearest_conv_upsample_256_w8_16_v4_full.csv",
}


def num(v, default=0.0):
    return float(v) if v not in (None, "") else default


def load_threshold_rows(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["node_kind"] == "Thresholding"]
    out = []
    for r in rows:
        pe = int(num(r["PE"], 1)) or 1
        channels = int(num(r["NumChannels"], 1)) or 1
        num_steps = num(r["numSteps"], 255)
        real_bram = num(r["real_BRAM18"]) + 2 * num(r["real_BRAM36"])
        real_lutram = num(r["real_LUTRAM"])
        out.append({"pe": pe, "channels": channels, "num_steps": num_steps, "real_bram": real_bram,
                     "real_lutram": real_lutram, "nonzero": real_bram > 0})
    return out


def best_threshold_split(rows, key_fn):
    """Scan every observed key value as a candidate cutoff (classify >= cutoff as
    'predict nonzero'); return (cutoff, n_misclassified, confusion) at the best split."""
    keys = sorted({key_fn(r) for r in rows})
    best = None
    for cutoff in keys:
        fp = fn = 0  # false positive: predicted nonzero, real zero. false negative: predicted zero, real nonzero.
        for r in rows:
            predicted_nonzero = key_fn(r) >= cutoff
            if predicted_nonzero and not r["nonzero"]:
                fp += 1
            elif not predicted_nonzero and r["nonzero"]:
                fn += 1
        total_wrong = fp + fn
        if best is None or total_wrong < best[1]:
            best = (cutoff, total_wrong, fp, fn)
    return best


def ols_through_origin_1term(xy):
    sxx = sxy = 0.0
    for x, y in xy:
        sxx += x * x
        sxy += x * y
    return sxy / sxx if sxx else float("nan")


def r2_uncentered(xy, k):
    ss_res = ss_tot = 0.0
    for x, y in xy:
        ss_res += (y - k * x) ** 2
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

    per_build = {name: load_threshold_rows(path) for name, path in DATASETS.items()}
    pooled = [r for rows in per_build.values() for r in rows]

    print(f"{'build':8s} {'n':>4s} {'nonzero BRAM':>13s} {'zero (BRAM & LUTRAM)':>22s} {'nonzero LUTRAM':>15s}")
    for name, rows in per_build.items():
        nonzero = sum(r["nonzero"] for r in rows)
        zero_both = sum(not r["nonzero"] and r["real_lutram"] == 0 for r in rows)
        nonzero_lutram = sum(r["real_lutram"] > 0 for r in rows)
        print(f"{name:8s} {len(rows):4d} {nonzero:13d} {zero_both:22d} {nonzero_lutram:15d}")

    print("\nCandidate cutoff variables (best single-threshold split, pooled v2+v4):")
    candidates = {
        "PE*numSteps": lambda r: r["pe"] * r["num_steps"],
        "NumChannels*numSteps": lambda r: r["channels"] * r["num_steps"],
        "PE*NumChannels*numSteps": lambda r: r["pe"] * r["channels"] * r["num_steps"],
    }
    results = {}
    for label, key_fn in candidates.items():
        cutoff, wrong, fp, fn = best_threshold_split(pooled, key_fn)
        results[label] = (cutoff, wrong, fp, fn, key_fn)
        print(f"  {label:26s} cutoff={cutoff:10.1f}  misclassified={wrong:4d}/{len(pooled)} "
              f"(false-positive={fp}, false-negative={fn})")

    best_label = min(results, key=lambda k: results[k][1])
    cutoff, wrong, fp, fn, key_fn = results[best_label]
    print(f"\nBest: {best_label} @ cutoff={cutoff:.1f} ({wrong}/{len(pooled)} misclassified, "
          f"{100 * wrong / len(pooled):.1f}%) -- NOT a clean boundary, treat as a soft empirical gate.")

    above_cutoff = [r for r in pooled if key_fn(r) >= cutoff]
    xy_old = [(r["pe"] * r["num_steps"], r["real_bram"]) for r in above_cutoff]
    k_new = ols_through_origin_1term(xy_old)
    r2_new = r2_uncentered(xy_old, k_new)
    abs_pct_errs = [abs(k_new * x - y) / y * 100 for x, y in xy_old if y]
    print(f"\nRefit _THR_RTL_BRAM18_PER_PE_NUMSTEP on the {len(above_cutoff)} above-cutoff rows only "
          f"(1-term OLS through origin, real_bram ~= k*PE*numSteps):")
    print(f"  k = {k_new:.6f}  (current live rate: {LIVE_RATE}, live cutoff: {LIVE_CUTOFF})")
    print(f"  R^2 (uncentered) = {r2_new:.4f}   n = {len(above_cutoff)}   "
          f"median abs %% err (per node) = {median(abs_pct_errs):.1f}%%")

    print("\nPer-build aggregate threshold BRAM error, CURRENT LIVE (rate+cutoff) vs THIS RUN's fresh refit:")
    per_build_err = {}
    for name, rows in per_build.items():
        real_sum = sum(r["real_bram"] for r in rows)
        live_pred = sum(
            LIVE_RATE * r["pe"] * r["num_steps"] if r["channels"] * r["num_steps"] >= LIVE_CUTOFF else 0.0
            for r in rows
        )
        new_pred = sum(k_new * r["pe"] * r["num_steps"] if key_fn(r) >= cutoff else 0.0 for r in rows)
        live_err = 100 * (live_pred - real_sum) / real_sum
        new_err = 100 * (new_pred - real_sum) / real_sum
        per_build_err[name] = (live_err, new_err)
        print(f"  {name:8s} live_err={live_err:+7.1f}%   new_fit_err={new_err:+7.1f}%")

    print(f"\nSuggested finn_cost_model.py constants:")
    print(f"  _THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP = {cutoff:.1f}   (variable: {best_label})")
    print(f"  _THR_RTL_BRAM18_PER_PE_NUMSTEP = {k_new:.6f}   (refit, above-cutoff subset only)")

    if args.write_calibration_csv:
        row = {
            "date": date.today().isoformat(),
            "component": "threshold_bram",
            "constant": "_THR_RTL_BRAM_MIN_CHANNELS_NUMSTEP / _THR_RTL_BRAM18_PER_PE_NUMSTEP",
            "old_value": f"cutoff={LIVE_CUTOFF} / rate={LIVE_RATE} (live before this run)",
            "new_value": f"cutoff={cutoff:.1f} ({best_label}) / rate={k_new:.6f}",
            "method": "Decision-stump misclassification scan over PE*numSteps / NumChannels*numSteps / "
                      "PE*NumChannels*numSteps (cutoff), then 1-term OLS through origin on the above-cutoff "
                      "subset only (rate), via hardware/fit_threshold_bram_cutoff.py",
            "dataset": "12_dense_relu_nearest_conv_upsample_256_v2 + _w8_16_v4 ("
                       + "; ".join(f"{n}: {len(r)} rows/{sum(x['nonzero'] for x in r)} nonzero"
                                    for n, r in per_build.items()) + ")",
            "n": f"{len(above_cutoff)} (above-cutoff subset)",
            "r2_uncentered": f"{r2_new:.4f}",
            "median_abs_pct_err": f"{median(abs_pct_errs):.1f}%",
            "status": "applied_partial",
            "notes": f"{wrong}/{len(pooled)} ({100 * wrong / len(pooled):.1f}%) misclassified at the chosen "
                     f"cutoff -- NOT a clean boundary. Per-build error, live before this run vs this run's fresh "
                     f"refit: " + "; ".join(f"{name} {live:+.1f}%->{new:+.1f}%" for name, (live, new) in per_build_err.items())
                     + ". real PE degenerate at 1 in both builds (folding-config-propagation bug, see "
                       "threshold_lut rows), so a pure PE*numSteps cutoff carries no signal.",
        }
        append_calibration_row(row)


if __name__ == "__main__":
    main()
