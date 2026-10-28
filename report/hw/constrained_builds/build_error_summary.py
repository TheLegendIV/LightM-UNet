"""Percent error of the cost model against real out-of-context synthesis, for the four constrained S12-512 designs (INT4, INT6 MILP fold -- not the FINN auto-fold build --, INT8, Mixed 4/6/8):
rows = LUT, BRAM18, DSP (cycles: see below), columns = uncalibrated (stock FINN estimators) and calibrated (this repo's cost model), both after the FIFO simulation.
error % = 100 * (model - real) / real, so + = overestimate, - = underestimate. The aggregate is the unweighted MEAN of the four SIGNED errors (the mean absolute error is written next to it).

Reads modelled_vs_real_cost.csv (plot_constrained_builds.py); writes model_error_summary.csv (aggregate) and model_error_by_design.csv (every design).
Cycles: no measured cycles exist for these four builds (no board / full-network rtlsim run), so the row is left empty rather than compared with a simulation.
    python report/MILP/constrained_builds/build_error_summary.py
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = {"LUT": ("lut", "model_lut", "model_uncal_lut", "real_lut"), "BRAM18": ("bram", "model_bram18", "model_uncal_bram18", "real_bram18eq"), "DSP": ("dsp", "model_dsp", "model_uncal_dsp", "real_dsp")}


def main() -> int:
    data = list(csv.DictReader(open(HERE / "modelled_vs_real_cost.csv", newline="")))
    schemes = [r["scheme"] for r in data]
    err = lambda model, real: 100.0 * (float(model) - float(real)) / float(real)      # noqa: E731
    by_design, summary = [], []
    for name, (_, cal_col, unc_col, real_col) in ROWS.items():
        cal = [err(r[cal_col], r[real_col]) for r in data]
        unc = [err(r[unc_col], r[real_col]) for r in data]
        by_design.append([name, "uncalibrated"] + [f"{e:+.1f}" for e in unc] + [f"{sum(unc) / len(unc):+.1f}"])
        by_design.append([name, "calibrated"] + [f"{e:+.1f}" for e in cal] + [f"{sum(cal) / len(cal):+.1f}"])
        summary.append([name, f"{sum(unc) / len(unc):+.1f}", f"{sum(cal) / len(cal):+.1f}", f"{sum(abs(e) for e in unc) / len(unc):.1f}", f"{sum(abs(e) for e in cal) / len(cal):.1f}"])
    summary.append(["Cycles", "", "", "", ""])
    with open(HERE / "model_error_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resource", "uncalibrated_mean_error_pct", "calibrated_mean_error_pct", "uncalibrated_mean_abs_error_pct", "calibrated_mean_abs_error_pct"])
        w.writerows(summary)
    with open(HERE / "model_error_by_design.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resource", "model"] + schemes + ["mean"])
        w.writerows(by_design)
    for path in ("model_error_summary.csv", "model_error_by_design.csv"):
        print(open(HERE / path).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
