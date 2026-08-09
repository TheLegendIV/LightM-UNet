"""Final proof-of-beat plot: stage_1_naive_baseline's own channel-width
curve (Base/U2/U4/U8/U16) vs. S21 (S19's dense_dilation_reg_interleaved_
double_mid architecture swept across the same widths) vs. S19 itself
(the architecture's own U4-width point) -- for all three cost metrics
(params, MACs, FINN buffer memory elements). Direct answer to "does S19's
architecture genuinely beat the naive width-compression curve, not just at
one param count": stage_21_reginterleaved_separable_nonneg_block_double_mid_width_array.job's
own motivation.

Only includes stage_1_naive_baseline / stage_21_.../ S19 rows -- everything
else in results.csv (arch probes, quantization experiments, pruning, etc.)
is deliberately excluded, unlike plot_all_configs.py's everything-at-once
view.

Usage:
    python compression/plot_final_width_comparison.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
NAIVE_COLOR = "#2a78d6"
CURVE_COLOR = "#c3392b"
S21_COLOR = "#8e24aa"
S19_COLOR = "#f4511e"

NAIVE_STAGE = "1_naive_baseline"
S21_STAGE = "21_reginterleaved_separable_nonneg_block_double_mid_width"
S19_CONFIG_NAME = "nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid"

NAIVE_ABBREV = {
    "nnUNetTrainerENet_1_naive_baseline_Baseline": "Base",
    "nnUNetTrainerENet_1_naive_baseline_U2": "U2",
    "nnUNetTrainerENet_1_naive_baseline_U3": "U3",
    "nnUNetTrainerENet_1_naive_baseline_U4": "U4",
    "nnUNetTrainerENet_1_naive_baseline_U6": "U6",
    "nnUNetTrainerENet_1_naive_baseline_U8": "U8",
    "nnUNetTrainerENet_1_naive_baseline_U16": "U16",
    "nnUNetTrainerENet_1_naive_baseline_E1": "E1",
}
S21_ABBREV = {
    "nnUNetTrainerENet_21_1_u2": "S21-U2",
    "nnUNetTrainerENet_21_2_u8": "S21-U8",
    "nnUNetTrainerENet_21_3_original": "S21-Original",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)
    ax.set_xscale("log")


def _plot_metric(naive: pd.DataFrame, s21: pd.DataFrame, s19: pd.DataFrame,
                  x_col: str, x_label: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    _style_axes(ax)

    # PCHIP curve through the naive baseline's own points, sorted by x --
    # same monotone-cubic choice as plot_all_configs.py's own curve (never
    # overshoots the data's own local trend, unlike a raw polynomial fit).
    naive_sorted = naive.dropna(subset=[x_col, "dice"]).sort_values(x_col)
    if len(naive_sorted) >= 3:
        log_x = np.log10(naive_sorted[x_col].to_numpy(dtype=float))
        y = naive_sorted["dice"].to_numpy(dtype=float)
        interpolator = PchipInterpolator(log_x, y)
        dense_log_x = np.linspace(log_x.min(), log_x.max(), 200)
        ax.plot(10 ** dense_log_x, interpolator(dense_log_x), color=CURVE_COLOR, linewidth=1.5,
                 zorder=2, label="naive width-compression curve (PCHIP)")

    ax.scatter(naive_sorted[x_col], naive_sorted["dice"], color=NAIVE_COLOR, s=90, zorder=3,
               edgecolors="black", linewidths=0.6, label="1_naive_baseline")
    for _, row in naive_sorted.iterrows():
        label = NAIVE_ABBREV.get(row["config_name"], row["config_name"])
        ax.annotate(label, (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                   textcoords="offset points", xytext=(6, 4))

    s21_sorted = s21.dropna(subset=[x_col, "dice"]).sort_values(x_col)
    if not s21_sorted.empty:
        ax.scatter(s21_sorted[x_col], s21_sorted["dice"], color=S21_COLOR, s=110, zorder=4,
                   marker="D", edgecolors="black", linewidths=0.7, label="S21 (S19 arch, width-swept)")
        for _, row in s21_sorted.iterrows():
            label = S21_ABBREV.get(row["config_name"], row["config_name"])
            ax.annotate(label, (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                       textcoords="offset points", xytext=(6, -12))

    s19_sorted = s19.dropna(subset=[x_col, "dice"])
    if not s19_sorted.empty:
        ax.scatter(s19_sorted[x_col], s19_sorted["dice"], color=S19_COLOR, s=180, zorder=5,
                   marker="*", edgecolors="black", linewidths=0.8, label="S19 (own U4 width)")
        for _, row in s19_sorted.iterrows():
            ax.annotate("S19", (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                       textcoords="offset points", xytext=(8, 6))

    ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
    ax.set_title(f"Naive width curve vs. S19/S21: Dice vs. {x_label}", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1
    df = pd.read_csv(args.results_csv)

    naive = df[df["stage"] == NAIVE_STAGE]
    s21 = df[df["stage"] == S21_STAGE]
    s19 = df[df["config_name"] == S19_CONFIG_NAME]
    if naive.empty:
        print(f"No {NAIVE_STAGE} rows found.")
        return 1

    df = df.copy()
    df["macs"] = df["flops"] / 2
    naive = naive.copy()
    naive["macs"] = naive["flops"] / 2
    s21 = s21.copy()
    s21["macs"] = s21["flops"] / 2
    s19 = s19.copy()
    s19["macs"] = s19["flops"] / 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(naive, s21, s19, "params", "Parameters", args.out_dir / "dice_vs_params_final.png")
    _plot_metric(naive, s21, s19, "macs", "MACs", args.out_dir / "dice_vs_macs_final.png")
    _plot_metric(naive, s21, s19, "mem_elements", "FINN buffer memory (activation elements)",
                 args.out_dir / "dice_vs_mem_elements_final.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
