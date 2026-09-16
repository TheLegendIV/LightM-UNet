"""Refits finn_cost_model.py's `imbalance_luts` correction term against
hardware/mvau_lut_calibration_dataset.csv (89 real MVAU/VVAU nodes, 8
partitions, one force_dsp=True S12 dense build), using a GATED QUADRATIC
form instead of the current linear/unconditional one:

    OLD (current, in finn_cost_model.py):
        gap = max(0, PE - SIMD)
        imbalance_luts = c_imbalance * gap * A + c_mh * MH        (MH unconditional)

    NEW (this script fits):
        gap = max(0, PE - SIMD)
        imbalance_luts = (c1 * A + c2 * MH) * gap ** 2            (MH gated by gap too)

Motivation: the OLD form's `c_mh * MH` term fires on EVERY layer regardless
of PE/SIMD balance -- confirmed empirically to badly overshoot a SEPARABLE
build (min4 hardcap131: raw LUT prediction jumped from 1.17x-under to
4.3x-OVER real, almost entirely from this unconditional term, since
separable's own PE-SIMD gaps are mostly 0). Gating MH by gap**2 makes the
WHOLE correction vanish when PE<=SIMD (57/89 rows here), matching the
observation that balanced folding doesn't need this correction at all, while
still growing superlinearly for large gaps (up to 31 in this dataset).

baseline_lut (everything in conv_cost_pe_simd's mvu_lut EXCEPT
imbalance_luts) is computed directly from this CSV's own MH/MW/PE/SIMD/bits
columns -- no LayerGeometry reconstruction needed, since addertree_luts/
acc_luts/swu_lut depend only on those:
    addertree_luts = (W+A)*(2*SIMD-1)
    acc_luts        = min(32, alpha + log2(1+2**-alpha) + 1), alpha=log2(MW)+W+A-2
    mvu_lut_baseline = 300 + 1.1*PE*(addertree_luts + acc_luts)   (mult_luts=0, force_dsp=True for every row)
    baseline_lut     = 426 + mvu_lut_baseline                      (swu_lut=426, M=1)

Fits (c1, c2) via ordinary least squares, NO intercept (forced through the
origin, same convention as the existing fit -- gap=0 rows must predict
exactly baseline_lut with zero residual left unexplained by design, not
absorbed into a free intercept).

Usage:
    python compression/hawq/fit_imbalance_luts_quadratic.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO_ROOT / "hardware" / "mvau_lut_calibration_dataset.csv"


def baseline_lut(pe: int, simd: int, w: int, a: int, mw: int) -> float:
    addertree_luts = (w + a) * (2 * simd - 1)
    alpha = math.log2(mw) + w + a - 2
    acc_luts = min(32, alpha + math.log2(1 + 2 ** -alpha) + 1)
    mvu_lut = 300 + 1.1 * pe * (addertree_luts + acc_luts)
    return 426 + mvu_lut


def old_imbalance(pe: int, simd: int, a: int, mh: int, c_imbalance: float = 167.38, c_mh: float = 274.14) -> float:
    gap = max(0, pe - simd)
    return c_imbalance * gap * a + c_mh * mh


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["gap"] = (df["PE"] - df["SIMD"]).clip(lower=0)
    df["baseline"] = df.apply(lambda r: baseline_lut(r.PE, r.SIMD, r.weight_bits, r.act_bits, r.MW), axis=1)
    df["residual"] = df["real_LUT"] - df["baseline"]
    df["old_imbalance"] = df.apply(lambda r: old_imbalance(r.PE, r.SIMD, r.act_bits, r.MH), axis=1)
    df["old_pred"] = df["baseline"] + df["old_imbalance"]

    # New gated-quadratic fit: residual ~ c1*(gap^2 * A) + c2*(gap^2 * MH), no intercept.
    x1 = (df["gap"] ** 2) * df["act_bits"]
    x2 = (df["gap"] ** 2) * df["MH"]
    X = np.column_stack([x1, x2])
    y = df["residual"].to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c1, c2 = coeffs
    df["new_imbalance"] = c1 * x1 + c2 * x2
    df["new_pred"] = df["baseline"] + df["new_imbalance"]

    # HYBRID fit: keep MH UNCONDITIONAL (it explains real gap=0 variance --
    # see gap=0 R^2 collapsing to baseline-alone once gated away), only
    # square the gap-dependent piece. 2 regressors: MH (unconditional) and
    # gap^2*A (replaces the old LINEAR gap*A term).
    x3 = df["MH"]
    x4 = (df["gap"] ** 2) * df["act_bits"]
    X2 = np.column_stack([x3, x4])
    coeffs2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
    c_mh2, c_gap2 = coeffs2
    df["hybrid_imbalance"] = c_mh2 * x3 + c_gap2 * x4
    df["hybrid_pred"] = df["baseline"] + df["hybrid_imbalance"]
    print(f"Hybrid fit (MH unconditional + gap^2*A): imbalance_luts = {c_mh2:.4f}*MH + {c_gap2:.4f}*gap**2*A")

    def r_squared(pred, actual):
        ss_res = ((actual - pred) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    def rmse(pred, actual):
        return float(np.sqrt(((actual - pred) ** 2).mean()))

    print(f"n={len(df)} rows, {len(df[df.gap == 0])} at gap=0, {len(df[df.gap > 0])} at gap>0")
    print()
    print(f"Fitted: imbalance_luts = ({c1:.4f}*A + {c2:.4f}*MH) * gap**2")
    print()
    print("=== Whole-dataset fit quality ===")
    print(f"{'':20s} {'R^2':>8s} {'RMSE':>10s}")
    print(f"{'baseline alone':20s} {r_squared(df['baseline'], df['real_LUT']):8.3f} {rmse(df['baseline'], df['real_LUT']):10.1f}")
    print(f"{'OLD (linear+MH)':20s} {r_squared(df['old_pred'], df['real_LUT']):8.3f} {rmse(df['old_pred'], df['real_LUT']):10.1f}")
    print(f"{'NEW (gated quad)':20s} {r_squared(df['new_pred'], df['real_LUT']):8.3f} {rmse(df['new_pred'], df['real_LUT']):10.1f}")
    print(f"{'HYBRID (MH+quad)':20s} {r_squared(df['hybrid_pred'], df['real_LUT']):8.3f} {rmse(df['hybrid_pred'], df['real_LUT']):10.1f}")

    print()
    print("=== Split by PE<=SIMD (gap=0, should need ~no correction) vs PE>SIMD (gap>0) ===")
    for label, mask in [("gap=0 (57 rows)", df.gap == 0), ("gap>0 (32 rows)", df.gap > 0)]:
        sub = df[mask]
        print(f"\n-- {label} --")
        print(f"  baseline alone : R^2={r_squared(sub['baseline'], sub['real_LUT']):7.3f}  RMSE={rmse(sub['baseline'], sub['real_LUT']):9.1f}")
        print(f"  OLD prediction : R^2={r_squared(sub['old_pred'], sub['real_LUT']):7.3f}  RMSE={rmse(sub['old_pred'], sub['real_LUT']):9.1f}")
        print(f"  NEW prediction : R^2={r_squared(sub['new_pred'], sub['real_LUT']):7.3f}  RMSE={rmse(sub['new_pred'], sub['real_LUT']):9.1f}")
        print(f"  HYBRID pred    : R^2={r_squared(sub['hybrid_pred'], sub['real_LUT']):7.3f}  RMSE={rmse(sub['hybrid_pred'], sub['real_LUT']):9.1f}")

    print()
    print("=== Power sweep on the GATED form: imbalance_luts = (c1*A + c2*MH) * gap**n ===")
    print(f"{'n':>3s} {'c1':>10s} {'c2':>10s} {'R^2 (all)':>10s} {'R^2 (gap>0)':>12s} {'RMSE (gap>0)':>13s}")
    gap_gt0 = df["gap"] > 0
    for n in (1, 2, 3, 4, 5, 6):
        xn1 = (df["gap"] ** n) * df["act_bits"]
        xn2 = (df["gap"] ** n) * df["MH"]
        Xn = np.column_stack([xn1, xn2])
        cn, _, _, _ = np.linalg.lstsq(Xn, y, rcond=None)
        pred_n = df["baseline"] + cn[0] * xn1 + cn[1] * xn2
        r2_all = r_squared(pred_n, df["real_LUT"])
        r2_gap = r_squared(pred_n[gap_gt0], df["real_LUT"][gap_gt0])
        rmse_gap = rmse(pred_n[gap_gt0], df["real_LUT"][gap_gt0])
        print(f"{n:>3d} {cn[0]:>10.4f} {cn[1]:>10.4f} {r2_all:>10.3f} {r2_gap:>12.3f} {rmse_gap:>13.1f}")

    print()
    print("=== Power sweep, WEIGHTED (relative-error) fit: weight = 1/real_LUT^2 ===")
    print("(prevents the handful of huge-real_LUT high-gap rows from dominating the fit --")
    print(" lets a genuinely accelerating shape show up if the data supports one, rather")
    print(" than getting swamped by raw magnitude.)")
    print(f"{'n':>5s} {'c1':>10s} {'c2':>10s} {'R^2 (all)':>10s} {'R^2 (gap>0)':>12s} {'median |rel err| gap>0':>24s}")
    w = 1.0 / (df["real_LUT"].to_numpy() ** 2)
    sw = np.sqrt(w)
    for n in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12):
        xn1 = (df["gap"] ** n) * df["act_bits"]
        xn2 = (df["gap"] ** n) * df["MH"]
        Xn = np.column_stack([xn1 * sw, xn2 * sw])
        yn = y * sw
        cn, _, _, _ = np.linalg.lstsq(Xn, yn, rcond=None)
        pred_n = df["baseline"] + cn[0] * xn1 + cn[1] * xn2
        r2_all = r_squared(pred_n, df["real_LUT"])
        r2_gap = r_squared(pred_n[gap_gt0], df["real_LUT"][gap_gt0])
        rel_err = ((pred_n[gap_gt0] - df["real_LUT"][gap_gt0]) / df["real_LUT"][gap_gt0]).abs()
        print(f"{n:>5.1f} {cn[0]:>10.4f} {cn[1]:>10.4f} {r2_all:>10.3f} {r2_gap:>12.3f} {rel_err.median():>24.3f}")

    print()
    print("=== Per-row detail (sorted by gap) ===")
    cols = ["partition", "node_name", "PE", "SIMD", "gap", "act_bits", "MH", "MW", "real_LUT", "baseline", "old_pred", "new_pred"]
    print(df[cols].sort_values("gap").to_string(index=False))


if __name__ == "__main__":
    main()
