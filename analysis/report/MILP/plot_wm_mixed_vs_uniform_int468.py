"""Warm-started-checkpoint ("_wm") family: dice vs. resource consumption for
the per-layer joint ILP's mixed-precision solve ("Mixed") against the naive
uniform-quantization baseline (INT4/INT6/INT8, folding pinned to the SAME
per-layer structure as the mixed solve -- see MILP/artifacts/
S12_dense_nearest_upsample_wm_v1_uniform_samefolding/). Same house style as
compression/analysis/qat_results/plot_alpha025_vs_uniform_int48.py (reuses
its INK/SECONDARY_INK/SURFACE/ALPHA_COLORS/_style_axes) and
plot_uniform_vs_ilp_overlay.py (reuses its UNIFORM_COLOR), but for this repo's
own MILP-solve-artifact family the plots themselves live under
analysis/report/MILP/ rather than compression/analysis/qat_results/, per this
repo's directory convention (MILP outputs vs. written-up analysis).

Three figures, one script:
  1. LUT% on the x-axis  -> figures/12_dense_relu_nearest_upsample_wm_mixed_vs_uniform_int468_lut.png
  2. BRAM% on the x-axis -> figures/12_dense_relu_nearest_upsample_wm_mixed_vs_uniform_int468_bram.png
  3. DSP% on the x-axis  -> figures/12_dense_relu_nearest_upsample_wm_mixed_vs_uniform_int468_dsp.png
     (all four points land at the same ~5.79% DSP -- force-dsp pins every
     config to the same minimum forced DSP count regardless of bit-width at
     this budget, so this axis is expected to be uninformative/degenerate;
     plotted anyway since it was explicitly requested.)

All dice values are each config's checkpoint_best.pth row (the EMA-best
checkpoint actually deployed), per explicit user instruction -- NOT the
epoch-15 fixed snapshot plot_alpha025_vs_uniform_int48.py uses.

Resource percentages are hardcoded from the real MILP-solve artifacts (same
convention as plot_uniform_vs_ilp_overlay.py's own UNIFORM_LUT_PCT dict) --
see RESOURCE_PCT below for the exact sources:
  - Mixed:   MILP/artifacts/S12_dense_nearest_upsample_wm_v1/summary.csv
  - Uniform: MILP/artifacts/S12_dense_nearest_upsample_wm_v1_uniform_samefolding/
             layer_bits_folding_uniform_int{4,6,8}_samefolding.json's own
             _diagnostics block.

Usage:
    python analysis/report/MILP/plot_wm_mixed_vs_uniform_int468.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "compression" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import (  # noqa: E402
    INK, SECONDARY_INK, SURFACE, ALPHA_COLORS, REPO_ROOT, _style_axes,
)
from plot_uniform_vs_ilp_overlay import UNIFORM_COLOR  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "compression" / "results.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_LUT = OUT_DIR / "12_dense_relu_nearest_upsample_wm_mixed_vs_uniform_int468_lut.png"
OUT_BRAM = OUT_DIR / "12_dense_relu_nearest_upsample_wm_mixed_vs_uniform_int468_bram.png"
OUT_DSP = OUT_DIR / "12_dense_relu_nearest_upsample_wm_mixed_vs_uniform_int468_dsp.png"

FP32_CONFIG_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_upsample_warmstart150ep"

BEST_CONFIG_NAMES = {
    "Mixed": "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
             "12_dense_relu_nearest_upsample_wm_joint_alpha1.0_perlayer_candidatebits468_"
             "forcedsp_lut50_bram50_dsp90_ft15ep",
    4: "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
       "12_dense_relu_nearest_upsample_wm_uniform_int4_samefolding_perlayer_"
       "lut50_bram50_dsp90_ft15ep",
    6: "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
       "12_dense_relu_nearest_upsample_wm_uniform_int6_samefolding_perlayer_"
       "lut50_bram50_dsp90_ft15ep",
    8: "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
       "12_dense_relu_nearest_upsample_wm_uniform_int8_samefolding_perlayer_"
       "lut50_bram50_dsp90_ft15ep",
}

# Real numbers from the MILP-solve artifacts themselves (see module docstring
# for exact sources) -- not re-derived here since that requires the real
# cost-model call. Both budgets are a hard 50% cap for this solve family.
RESOURCE_PCT = {
    "lut": {"Mixed": 49.88, 4: 42.83, 6: 45.67, 8: 52.58},
    "bram": {"Mixed": 49.94, 4: 14.76, 6: 19.96, 8: 73.42},
    # All four land at essentially the same ~5.79% -- force-dsp pins every
    # config to the same minimum forced DSP count regardless of bit-width at
    # this budget (see MILP/artifacts/S12_dense_nearest_upsample_wm_v1/
    # summary.csv and the sibling uniform_samefolding JSONs' _diagnostics).
    "dsp": {"Mixed": 5.79, 4: 5.79, 6: 5.79, 8: 5.79},
}
# Each metric's own hard cap for this solve family (LUT/BRAM 50%, DSP 90%) --
# NOT all normalized to 100.
BUDGET_PCT = {"lut": 50.0, "bram": 50.0, "dsp": 90.0}


def load_dice_best(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = {row["config_name"]: row for row in csv.DictReader(f)}
    dice = {}
    for key, name in BEST_CONFIG_NAMES.items():
        if name not in rows:
            print(f"Note: config_name not found in {csv_path}: {name!r} (key={key!r}) -- skipping.")
            continue
        dice[key] = float(rows[name]["dice"])
    return dice


def load_fp32_baseline(csv_path: Path) -> float | None:
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["config_name"] == FP32_CONFIG_NAME:
                return float(row["dice"])
    return None


def plot_one(metric: str, axis_label: str, title: str, dice: dict, fp32_dice: float | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pct = RESOURCE_PCT[metric]

    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    _style_axes(ax)

    if fp32_dice is not None:
        ax.axhline(fp32_dice, color=INK, linestyle=":", linewidth=1.2, zorder=1,
                   label=f"FP32 baseline ({fp32_dice:.4f})")

    budget = BUDGET_PCT[metric]
    ax.axvline(budget, color=INK, linewidth=1.2, linestyle="-", zorder=1, alpha=0.6)

    # -- Mixed (per-layer joint ILP solve, alpha=1.0), checkpoint_best.pth. --
    if "Mixed" in dice:
        mixed_x, mixed_y = pct["Mixed"], dice["Mixed"]
        color = ALPHA_COLORS.get(1.0)
        ax.scatter([mixed_x], [mixed_y], color=color, s=130, zorder=5,
                   marker="o", edgecolor=SURFACE, linewidth=1.4)
        ax.annotate("Mixed", (mixed_x, mixed_y), xytext=(8, 8),
                    textcoords="offset points", color=color, fontsize=9)

    # -- Naive uniform baseline, INT4/INT6/INT8, folding pinned to the mixed
    # solve's own structure. Solid filled circles; red fill signals a point
    # that exceeds its own resource budget cap. --
    for bits in (4, 6, 8):
        if bits not in dice:
            continue
        y = dice[bits]
        x = pct[bits]
        fits = x <= budget
        color = UNIFORM_COLOR if fits else "#c0392b"
        ax.scatter([x], [y], color=color, s=130, zorder=5, linewidth=2,
                   marker="o", edgecolor=SURFACE)
        ax.annotate(f"INT{bits}", (x, y), xytext=(8, 8),
                    textcoords="offset points", color=color, fontsize=9)

    ax.set_xlabel(axis_label, color=INK, fontsize=10, fontweight="bold")
    ax.set_ylabel("Dice", color=INK, fontsize=10, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold")
    ax.set_ylim(top=0.8)
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7",
              fontsize=8.5, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    dice = load_dice_best(DEFAULT_CSV)
    fp32_dice = load_fp32_baseline(DEFAULT_CSV)
    if fp32_dice is None:
        print(f"Note: no FP32 baseline row ({FP32_CONFIG_NAME!r}) found in {DEFAULT_CSV} -- "
              f"plotting without the FP32 reference line.")

    print("\nDice (checkpoint_best.pth) used for all three plots:")
    for key in ("Mixed", 4, 6, 8):
        if key in dice:
            print(f"  {key!s:>6}: dice={dice[key]:.4f}  LUT%={RESOURCE_PCT['lut'][key]:.2f}  "
                  f"BRAM%={RESOURCE_PCT['bram'][key]:.2f}  DSP%={RESOURCE_PCT['dsp'][key]:.2f}")

    title = "ILP-Chosen Mixed Quantization vs. Naive Uniform Quantization at Various Bit Widths"
    plot_one("lut", "LUT Consumption (%)", title, dice, fp32_dice, OUT_LUT)
    plot_one("bram", "BRAM18K Consumption (%)", title, dice, fp32_dice, OUT_BRAM)
    plot_one("dsp", "DSP Consumption (%)", title, dice, fp32_dice, OUT_DSP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
