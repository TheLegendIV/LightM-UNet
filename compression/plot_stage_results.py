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

# Muted, consistent palette (matches the rest of compression/'s plots).
INK, SECONDARY_INK, MUTED, GRID, BASELINE_COLOR, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
STAGE_COLOR = "#2a78d6"
BASELINE_MARKER_COLOR = "#eb6834"

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
        # ENet is THE baseline (real, current reference everything else is
        # pruned from) -- E1 is a deprecated/superseded reference, kept only
        # as a sanity check that new configs aren't regressing toward it.
        # Visually distinct so they don't read as co-equal.
        enet_df = baseline_df[baseline_df["config_name"] == "nnUNetTrainerENet_Original"]
        e1_df = baseline_df[baseline_df["config_name"] != "nnUNetTrainerENet_Original"]
        if not enet_df.empty:
            ax.scatter(enet_df[x_col], enet_df["dice"], color=BASELINE_MARKER_COLOR, marker="D",
                       s=90, zorder=4, label="ENet (baseline)", edgecolors="black", linewidths=0.5)
        if not e1_df.empty:
            ax.scatter(e1_df[x_col], e1_df["dice"], color=MUTED, marker="x",
                       s=70, zorder=4, label="E1 (deprecated)", linewidths=1.5)
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
        # These stages' own configs (O4-derived, ~25-27K params) sit at
        # ~1/15th of Original/E1's scale (~370-470K) -- on a linear axis
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
    # stage1 itself also gets its own plot (Original vs E1), with no
    # separate baseline overlay (it IS the baseline).
    plot_stage(BASELINE_STAGE, baseline_df, pd.DataFrame(columns=df.columns), args.out_dir)
    for stage in sorted(stages):
        stage_df = df[df["stage"] == stage]
        plot_stage(stage, stage_df, baseline_df, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
