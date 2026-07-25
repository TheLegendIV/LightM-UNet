"""Per-stage comparison plots from compression/results.csv: Dice vs Params
and Dice vs FLOPs, one pair of figures per distinct `stage` value. Every
stage's plot also shows the stage1 baselines (Original, E1) as reference
points, so each ablation/sweep stage is visually anchored against the
config it's actually being compared to.

Usage:
    python compression/plot_stage_results.py
    python compression/plot_stage_results.py --results-csv compression/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_STAGE = "stage1"
COMPRESSED_BASELINE_STAGE = "1a_seed"
COMPRESSED_BASELINE_CONFIG = "nnUNetTrainerENet_1a_seed_O4_s0"

# Muted, consistent palette (matches the rest of compression/'s plots).
INK, SECONDARY_INK, MUTED, GRID, BASELINE_COLOR, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
STAGE_COLOR = "#2a78d6"
BASELINE_MARKER_COLOR = "#eb6834"
COMPRESSED_BASELINE_MARKER_COLOR = "#8e5fd4"

# Fixed per-config marker/color for every baseline overlay point -- keyed by
# config_name (not "is it ENet") so a third reference (Compressed Baseline)
# slots in without the ENet-vs-everything-else split breaking. Every stage
# past 1a_seed is O4-derived, so 1a_seed_O4_s0 ("Compressed Baseline") is
# the reference every 1b/1c/2a/2b/1d ablation/sweep is actually judged
# against -- ENet/E1 stay in the same overlay as the uncompressed anchor.
BASELINE_STYLES = {
    "nnUNetTrainerENet_Original": dict(
        marker="D", color=BASELINE_MARKER_COLOR, s=90, label="ENet (baseline)",
        edgecolors="black", linewidths=0.5,
    ),
    "nnUNetTrainerENet_E1": dict(
        marker="x", color=MUTED, s=70, label="E1 (deprecated)", linewidths=1.5,
    ),
    COMPRESSED_BASELINE_CONFIG: dict(
        marker="^", color=COMPRESSED_BASELINE_MARKER_COLOR, s=100, label="Compressed Baseline (O4)",
        edgecolors="black", linewidths=0.5,
    ),
}

# 2a's grid crosses 5 filter widths x 3 bottleneck patterns -- filter width
# already separates points along the x-axis (params/FLOPs), so bottleneck
# pattern (the axis this session just redesigned: native vs the new sparse
# dilation-4/16 div2/div4 patterns) gets categorical color instead, fixed
# order, never cycled.
BNECK_PATTERN_COLORS = {"native": "#2a78d6", "div2": "#eb6834", "div4": "#1baf7a"}


def bneck_pattern_from_config_name(config_name: str) -> str | None:
    for pattern in BNECK_PATTERN_COLORS:
        if config_name.endswith(f"_{pattern}"):
            return pattern
    return None

# Display-only renames for plot labels (results.csv's config_name / the
# underlying checkpoints are untouched). "Original" is ENet exactly as
# specified in the ENet paper -- the uncompressed baseline everything else
# is pruned from. 1a_seed_O4_s0 is O4 at native ops/no ablation applied --
# the reference point section 2a's pruning grid and every 1a/1b/1c ablation
# in this stage is compared against -- i.e. the "Compressed Baseline".
DISPLAY_NAME_OVERRIDES = {
    "nnUNetTrainerENet_Original": "ENet",
    "nnUNetTrainerENet_1a_seed_O4_s0": "Compressed Baseline",
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
        # cycled) so ENet/E1/Compressed Baseline stay visually distinct and
        # don't read as co-equal -- see BASELINE_STYLES.
        for config_name, style in BASELINE_STYLES.items():
            subset = baseline_df[baseline_df["config_name"] == config_name]
            if subset.empty:
                continue
            ax.scatter(subset[x_col], subset["dice"], zorder=4, **style)
        for _, row in baseline_df.iterrows():
            ax.annotate(display_name(row["config_name"]), (row[x_col], row["dice"]),
                       fontsize=7, color=SECONDARY_INK, textcoords="offset points", xytext=(5, 5))

    patterns = df["config_name"].map(bneck_pattern_from_config_name)
    has_pattern_groups = patterns.notna().any()
    if has_pattern_groups:
        # Color follows the entity (bottleneck pattern), not plot order --
        # fixed BNECK_PATTERN_COLORS assignment, legend below.
        for pattern, color in BNECK_PATTERN_COLORS.items():
            subset = df[patterns == pattern]
            if subset.empty:
                continue
            ax.scatter(subset[x_col], subset["dice"], color=color, s=80, zorder=3,
                       edgecolors="black", linewidths=0.5, label=pattern)
    else:
        ax.scatter(df[x_col], df["dice"], color=STAGE_COLOR, s=80, zorder=3,
                   edgecolors="black", linewidths=0.5)
    for _, row in df.iterrows():
        label = display_name(row["config_name"])
        ax.annotate(label, (row[x_col], row["dice"]), fontsize=7, color=SECONDARY_INK,
                   textcoords="offset points", xytext=(5, -9))

    ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Dice", color=SECONDARY_INK, fontsize=10)
    if not baseline_df.empty:
        # These stages' own configs (O4-derived, ~25-27K params) sit at
        # ~1/15th of Original/E1's scale (~370-470K) -- on a linear axis
        # they visually collapse onto the y-axis, indistinguishable from
        # each other, even though their Dice differs meaningfully. Log
        # scale keeps both clusters legible at once.
        ax.set_xscale("log")
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    if has_pattern_groups or not baseline_df.empty:
        # One legend call covering everything labeled so far (baseline
        # overlay + bottleneck-pattern groups, whichever are present) --
        # calling legend() more than once just replaces it with a fresh one
        # built from the same labeled artists, so consolidating avoids
        # redundant work and an inaccurate "bottleneck pattern" title when
        # ENet/E1 entries are mixed in too.
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
    compressed_baseline_df = df[df["config_name"] == COMPRESSED_BASELINE_CONFIG]
    stages = [s for s in df["stage"].unique() if s != BASELINE_STAGE]
    # stage1 itself also gets its own plot (Original vs E1), with no
    # separate baseline overlay (it IS the baseline).
    plot_stage(BASELINE_STAGE, baseline_df, pd.DataFrame(columns=df.columns), args.out_dir)
    for stage in sorted(stages):
        stage_df = df[df["stage"] == stage]
        # 1a_seed's own stage_df already contains the Compressed Baseline
        # row (1a_seed_O4_s0 is one of its three seeds) -- overlaying it
        # again would just duplicate that exact point. Every other O4-
        # derived stage (1b/1c/2a/2b/1d/...) gets it added to the stage1
        # (ENet/E1) overlay, per BASELINE_STYLES.
        if stage == COMPRESSED_BASELINE_STAGE:
            overlay_df = baseline_df
        else:
            overlay_df = pd.concat([baseline_df, compressed_baseline_df], ignore_index=True)
        plot_stage(stage, stage_df, overlay_df, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
