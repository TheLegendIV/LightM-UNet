"""Per-stage comparison plots from compression/results.csv: Dice vs Params
and Dice vs FLOPs, one pair of figures per distinct `stage` value. Every
stage's plot also shows stage_1_naive_baseline's Baseline/U4 reference
points, so each ablation/probe stage is visually anchored against the
config it's actually being compared to.

For the 4-class objective, "dice" is the mean across LAD/RCA/LCX/LM (see
compression/collect_results.py) -- per-class dice_LAD/dice_RCA/dice_LCX/
dice_LM columns also exist in results.csv but aren't plotted here by
default; this script's job is the same top-line Dice-vs-cost view the
binary run used, not a per-class breakdown.

Usage:
    python compression/plot_stage_results.py
    python compression/plot_stage_results.py --results-csv compression/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_STAGE = "1_naive_baseline"
COMPRESSED_BASELINE_CONFIG = "nnUNetTrainerENet_1_naive_baseline_U4"

# Muted, consistent palette (matches the rest of compression/'s plots).
INK, SECONDARY_INK, MUTED, GRID, BASELINE_COLOR, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
STAGE_COLOR = "#2a78d6"
BASELINE_MARKER_COLOR = "#eb6834"
COMPRESSED_BASELINE_MARKER_COLOR = "#8e5fd4"

# Fixed per-config marker/color for every baseline overlay point -- keyed by
# config_name so both references slot in without an ENet-vs-everything-else
# split. Every stage past 1_naive_baseline is U4-derived, so
# nnUNetTrainerENet_1_naive_baseline_U4 ("compressed baseline") is the
# reference every 2_special_ops/3_transfer_original/4_arch_probes probe is
# actually judged against -- the full-width Baseline stays in the same
# overlay as the uncompressed anchor.
BASELINE_STYLES = {
    "nnUNetTrainerENet_1_naive_baseline_Baseline": dict(
        marker="D", color=BASELINE_MARKER_COLOR, s=90, label="Baseline (full width)",
        edgecolors="black", linewidths=0.5,
    ),
    COMPRESSED_BASELINE_CONFIG: dict(
        marker="^", color=COMPRESSED_BASELINE_MARKER_COLOR, s=100, label="U4 (compressed baseline)",
        edgecolors="black", linewidths=0.5,
    ),
}

# Display-only renames for plot labels (results.csv's config_name / the
# underlying checkpoints are untouched).
DISPLAY_NAME_OVERRIDES = {
    "nnUNetTrainerENet_1_naive_baseline_Baseline": "Baseline",
    COMPRESSED_BASELINE_CONFIG: "U4 (compressed baseline)",
}


def display_name(config_name: str) -> str:
    if config_name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[config_name]
    return config_name.replace("nnUNetTrainerENet_", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-stage Dice vs Params/FLOPs plots.")
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def _scatter(ax, df: pd.DataFrame, x_col: str, x_label: str, baseline_df: pd.DataFrame) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BASELINE_COLOR)
    ax.tick_params(colors=MUTED, labelsize=9)

    if not baseline_df.empty:
        # Each reference config gets its own fixed marker/color (never
        # cycled) so Baseline/U4 stay visually distinct and don't read as
        # co-equal -- see BASELINE_STYLES.
        for config_name, style in BASELINE_STYLES.items():
            subset = baseline_df[baseline_df["config_name"] == config_name]
            if subset.empty:
                continue
            ax.scatter(subset[x_col], subset["dice"], zorder=4, **style)
        for _, row in baseline_df.iterrows():
            ax.annotate(display_name(row["config_name"]), (row[x_col], row["dice"]),
                       fontsize=7, color=SECONDARY_INK, textcoords="offset points", xytext=(5, 5))

    ax.scatter(df[x_col], df["dice"], color=STAGE_COLOR, s=80, zorder=3,
               edgecolors="black", linewidths=0.5)
    for _, row in df.iterrows():
        label = display_name(row["config_name"])
        ax.annotate(label, (row[x_col], row["dice"]), fontsize=7, color=SECONDARY_INK,
                   textcoords="offset points", xytext=(5, -9))

    ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Dice", color=SECONDARY_INK, fontsize=10)
    if not baseline_df.empty:
        # stage_2/3/4's own U4-derived configs (~20-30K params) sit at a
        # small fraction of Baseline's scale (~370K+) -- on a linear axis
        # they visually collapse onto the y-axis, indistinguishable from
        # each other, even though their Dice differs meaningfully. Log
        # scale keeps both clusters legible at once.
        ax.set_xscale("log")
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="best")


def plot_stage(stage: str, df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    valid = df.dropna(subset=["dice"])
    if valid.empty:
        print(f"[{stage}] no rows with a parsed Dice -- skipping.")
        return

    # Dice vs Params
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    _scatter(ax, valid, "params", "Parameters", baseline_df)
    ax.set_title(f"Stage {stage}: Dice vs Parameters", color=INK, fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{stage}_dice_vs_params.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Dice vs FLOPs
    flops_valid = valid.dropna(subset=["flops"])
    if flops_valid.empty:
        print(f"[{stage}] no rows with parsed FLOPs -- skipping Dice vs FLOPs.")
        return
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    _scatter(ax, flops_valid, "flops", "FLOPs", baseline_df.dropna(subset=["flops"]))
    ax.set_title(f"Stage {stage}: Dice vs FLOPs", color=INK, fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{stage}_dice_vs_flops.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1

    df = pd.read_csv(args.results_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = df[df["stage"] == BASELINE_STAGE]
    stages = [s for s in df["stage"].unique() if s != BASELINE_STAGE]
    # 1_naive_baseline itself also gets its own plot (Baseline/U2/U4/U8/U16),
    # with no separate overlay (it IS the baseline -- Baseline and U4 are
    # already plain data points within it).
    plot_stage(BASELINE_STAGE, baseline_df, pd.DataFrame(columns=df.columns), args.out_dir)
    for stage in sorted(stages):
        stage_df = df[df["stage"] == stage]
        # Every stage past 1_naive_baseline is U4-derived -- overlay the
        # full 1_naive_baseline set (which already contains both Baseline
        # and U4, per BASELINE_STYLES) so each probe is visually anchored
        # against both references.
        plot_stage(stage, stage_df, baseline_df, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
