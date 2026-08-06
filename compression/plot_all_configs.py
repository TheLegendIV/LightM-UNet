"""Dice vs. MACs and Dice vs. Params across EVERY trained config at once
(not per-stage like plot_stage_results.py), colored by stage, with a smooth
PCHIP-interpolated curve drawn through stage_1_naive_baseline's own points
specifically (the channel-width sweep -- Baseline/U2/U4/U8/U16 -- is the one
series here that forms an actual capacity/accuracy tradeoff curve; every
other stage is a handful of one-off probes off a single U4 reference point,
not a swept axis, so a fitted curve through them wouldn't mean anything).

Compact per-config labels come from compression/config_abbreviations.csv
(abbrev <-> config_name, kept as the source of truth so plot labels and the
underlying config_name never drift apart) -- see that file to add more rows
as later stages (e.g. stage_4_arch_probes) get real results.

Usage:
    python compression/plot_all_configs.py
    python compression/plot_all_configs.py --results-csv compression/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent
ABBREV_CSV = Path(__file__).resolve().parent / "config_abbreviations.csv"

INK, SECONDARY_INK, MUTED, GRID, BASELINE_COLOR, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
STAGE_COLORS = {
    "1_naive_baseline": "#2a78d6", "2_special_ops": "#eb6834",
    "3_transfer_original": "#1baf7a", "4_arch_probes": "#eda100",
    "5_arch_probe_pairs": "#8e44ad", "6_dscnoprojdense_variants": "#e91e8c",
    "7_reginterleaved_shape_variants": "#00acc1", "8_reginterleaved_isolation": "#7cb342",
    "9_dsc_projected_dense_dilation": "#9c6b30",
}
CURVE_STAGE = "1_naive_baseline"
CURVE_COLOR = "#c3392b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def load_abbreviations() -> dict[str, str]:
    abbrev_df = pd.read_csv(ABBREV_CSV)
    return dict(zip(abbrev_df["config_name"], abbrev_df["abbrev"]))


def _scatter_all(ax, df: pd.DataFrame, x_col: str, x_label: str, abbrev: dict[str, str]) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BASELINE_COLOR)
    ax.tick_params(colors=MUTED, labelsize=9)

    plotted = df.dropna(subset=[x_col, "dice"])

    # Smooth curve through stage_1_naive_baseline's own points -- sort by x
    # so the interpolant traces the actual capacity/accuracy tradeoff
    # instead of zig-zagging in config_name/row order. Fit with PCHIP
    # (monotonic cubic Hermite interpolation) rather than a raw polynomial:
    # a degree-4 polynomial forced through only 5 points is prone to Runge's
    # phenomenon (wild overshoot/oscillation between points), while PCHIP
    # passes through every point exactly, stays smooth (C1), and never
    # overshoots the data's own local monotonic trend. Fit in log10(x)
    # space, not linear x, since the axis itself is log-scaled -- fitting
    # in linear x and merely plotting on a log axis would NOT look smooth
    # to the eye; fitting in log-x space makes the rendered curve smooth.
    curve_df = plotted[plotted["stage"] == CURVE_STAGE].sort_values(x_col)
    if len(curve_df) >= 3:
        log_x = np.log10(curve_df[x_col].to_numpy(dtype=float))
        y = curve_df["dice"].to_numpy(dtype=float)
        interpolator = PchipInterpolator(log_x, y)
        dense_log_x = np.linspace(log_x.min(), log_x.max(), 200)
        ax.plot(10 ** dense_log_x, interpolator(dense_log_x), color=CURVE_COLOR, linewidth=1.5,
                 zorder=2, label=f"{CURVE_STAGE} curve (PCHIP)")
    elif len(curve_df) == 2:
        ax.plot(curve_df[x_col], curve_df["dice"], color=CURVE_COLOR, linewidth=1.5,
                 zorder=2, label=f"{CURVE_STAGE} curve")

    for stage, color in STAGE_COLORS.items():
        subset = plotted[plotted["stage"] == stage]
        if subset.empty:
            continue
        ax.scatter(subset[x_col], subset["dice"], color=color, s=80, zorder=3,
                   edgecolors="black", linewidths=0.5, label=stage)

    for _, row in plotted.iterrows():
        label = abbrev.get(row["config_name"], row["config_name"])
        ax.annotate(label, (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                   textcoords="offset points", xytext=(5, -9))

    ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="best")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1

    df = pd.read_csv(args.results_csv)
    if df.empty:
        print(f"{args.results_csv} is empty -- nothing to plot.")
        return 1
    abbrev = load_abbreviations()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # Dice vs Params
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    _scatter_all(ax, df, "params", "Parameters", abbrev)
    ax.set_title("All configs: Dice vs. Parameters", color=INK, fontsize=12)
    fig.tight_layout()
    out_path = args.out_dir / "all_configs_dice_vs_params.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Dice vs MACs (flops column is 2*MACs, from utils.count_flops -- MACs = flops/2)
    df = df.copy()
    df["macs"] = df["flops"] / 2
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    _scatter_all(ax, df, "macs", "MACs", abbrev)
    ax.set_title("All configs: Dice vs. MACs", color=INK, fontsize=12)
    fig.tight_layout()
    out_path = args.out_dir / "all_configs_dice_vs_macs.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Dice vs FINN buffer-memory elements (utils.count_buffer_elements --
    # activation line-buffer size, not weight/params memory; see that
    # function's docstring). Only present for rows collected after this
    # column was added / backfilled -- dropna keeps older/partial rows from
    # plotting at mem_elements=NaN.
    if "mem_elements" in df.columns:
        mem_df = df.dropna(subset=["mem_elements"])
        if not mem_df.empty:
            fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
            _scatter_all(ax, mem_df, "mem_elements", "FINN buffer memory (activation elements)", abbrev)
            ax.set_title("All configs: Dice vs. FINN buffer memory", color=INK, fontsize=12)
            fig.tight_layout()
            out_path = args.out_dir / "all_configs_dice_vs_mem_elements.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
            plt.close(fig)
            print(f"Wrote {out_path}")

    # Print the abbreviation lookup + underlying numbers actually plotted,
    # matching every other script in this folder's "print what got written"
    # convention -- not just the two PNGs.
    cols = ["config_name", "stage", "params", "flops", "mem_elements", "dice"]
    cols = [c for c in cols if c in df.columns]
    printable = df[cols].copy()
    printable.insert(0, "abbrev", printable["config_name"].map(lambda c: abbrev.get(c, c)))
    print()
    print(printable.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
