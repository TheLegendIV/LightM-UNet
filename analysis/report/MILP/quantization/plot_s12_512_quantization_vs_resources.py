"""S12 dense bilinear-512: Dice of uniform INT4 / INT6 / INT8 and the mixed 6/8-bit allocation against the MILP's own resource estimates (NOT simulated: the `*_pct_of_budget` columns of
each arm's summary.csv, percent of the XCZU7EV). Same house style as compression/analysis/qat_results/plot_alpha025_vs_uniform_int48.py (reuses its colours and axes styling); one figure per resource:

  s12_512_quantization_lut.png   Dice vs LUT %
  s12_512_quantization_bram.png  Dice vs BRAM18K %
  s12_512_quantization_dsp.png   Dice vs DSP %

Resource estimates (MILP/artifacts/S12_dense_512_uniform_cap1_v1/<arm>/summary.csv, 250 FPS @ 100 MHz, force-DSP):
  uniform INT4 / INT6 / INT8: --min-resources, every cap 1.0;  mixed 6/8: accuracy objective, 0.5 caps (then a min-resources pass, --lexicographic).
INT8 needs 55.6 % of the LUTs, so it does not fit the 0.5 resource cap the mixed design is held to (red). Dice: the QAT evaluation table (FP32 is the dotted reference).

Usage:
    python analysis/report/MILP/quantization/plot_s12_512_quantization_vs_resources.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "compression" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, ALPHA_COLORS, _style_axes  # noqa: E402
from plot_uniform_vs_ilp_overlay import UNIFORM_COLOR  # noqa: E402

ARM_DIR = REPO_ROOT / "MILP" / "artifacts" / "S12_dense_512_uniform_cap1_v1"
ARMS = {4: "uniform_int4", 6: "uniform_int6", 8: "uniform_int8", "Mixed": "mixed_68"}
RESOURCE_CAP_PCT = 50.0        # the cap the mixed design is solved under (LUT / BRAM / DSP / URAM each)
INFEASIBLE_COLOR = "#c0392b"
MIXED_COLOR = ALPHA_COLORS.get(0.25)

FP32_DICE = 0.795
DICE = {8: 0.793, "Mixed": 0.786, 6: 0.771, 4: 0.643}      # the QAT evaluation table
FITS = {4: True, 6: True, 8: False, "Mixed": True}          # fits the 0.5 resource cap at 250 FPS

RESOURCES = {"lut": ("lut_pct_of_budget", "LUT Consumption (%)"),
             "bram": ("bram_pct_of_budget", "BRAM18K Consumption (%)"),
             "dsp": ("dsp_pct_of_budget", "DSP Consumption (%)")}


def load_pct(column: str) -> dict:
    out = {}
    for key, arm in ARMS.items():
        with open(ARM_DIR / arm / "summary.csv", newline="") as f:
            row = next(csv.DictReader(f))
        assert row["status"] == "Optimal", (arm, row["status"])
        out[key] = float(row[column])
    return out


def point_color(key) -> str:
    if key == "Mixed":
        return MIXED_COLOR
    return UNIFORM_COLOR if FITS[key] else INFEASIBLE_COLOR


def plot_one(resource: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    column, xlabel = RESOURCES[resource]
    x = load_pct(column)
    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    _style_axes(ax)
    ax.axhline(FP32_DICE, color=INK, linestyle=":", linewidth=1.2, zorder=1, label=f"FP32 baseline ({FP32_DICE:g})")
    ax.axvline(RESOURCE_CAP_PCT, color=INK, linewidth=1.2, linestyle="-", zorder=1, alpha=0.6, label="0.5 resource cap")
    for key in (4, 6, 8, "Mixed"):
        ax.scatter([x[key]], [DICE[key]], color=point_color(key), s=130, zorder=5, marker="o", edgecolor=SURFACE, linewidth=1.4)
        near_cap = abs(x[key] - RESOURCE_CAP_PCT) < 4                      # keep the label off the cap line: put it left of the point
        ax.annotate("Mixed 6/8" if key == "Mixed" else f"INT{key}", (x[key], DICE[key]), xytext=(-8 if near_cap else 8, 8), textcoords="offset points",
                    ha="right" if near_cap else "left", color=point_color(key), fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel("Dice", color=INK, fontsize=11, fontweight="bold")
    ax.set_title(f"Dice vs. {xlabel.split(' ')[0]} Consumption", color=INK, fontsize=12, fontweight="bold")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, RESOURCE_CAP_PCT + 3))
    ax.set_ylim(top=0.8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    leg = ax.legend(loc="best", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7", fontsize=9, labelcolor=SECONDARY_INK)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    print("MILP estimates (percent of the XCZU7EV, unsimulated):")
    for resource, (column, _) in RESOURCES.items():
        print(f"  {resource}: " + ", ".join(f"{k}={v:.2f}" for k, v in load_pct(column).items()))
    for resource in RESOURCES:
        plot_one(resource, HERE / f"s12_512_quantization_{resource}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
