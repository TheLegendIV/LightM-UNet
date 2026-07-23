"""Per-sweep progress PNG (agent_instructions_1.yaml top-of-file note: "Each
sweep should also render a png with all the configurations in the sweep as
a progress update"). Generic across stages -- reads compression/results.csv
filtered by --stage, plots whatever rows exist so far (safe to re-run mid-
sweep as an actual progress update, not just a final summary).

Axis choice: Dice vs. params by default (agent_instructions_1.yaml's
metrics.primary_gate=dice, efficiency_fp32=[params,flops]) -- the two
numbers Stage-1 goals are defined on. --x flops switches the second axis.

Usage:
    python compression/plot_sweep_progress.py stage1b
    python compression/plot_sweep_progress.py stage2 --x flops
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"
OUT_DIR = Path(__file__).resolve().parent / "sweep_progress"

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", help="Value of the 'stage' column to filter to, e.g. stage1b, stage2, stage2_4.")
    parser.add_argument("--x", default="params", choices=["params", "flops"], help="X-axis metric (default: params).")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        print(f"{RESULTS_CSV} does not exist yet -- nothing to plot.")
        return
    df = pd.read_csv(RESULTS_CSV)
    stage_df = df[df["stage"] == args.stage].dropna(subset=[args.x, "dice"])
    if stage_df.empty:
        print(f"No rows yet for stage={args.stage!r} (or all missing {args.x}/dice) -- nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.scatter(stage_df[args.x], stage_df["dice"], color=SERIES_BLUE, s=48, zorder=3, edgecolors=SURFACE, linewidths=0.5)
    for _, row in stage_df.iterrows():
        ax.annotate(str(row["config_name"]), (row[args.x], row["dice"]), fontsize=7,
                    color=SECONDARY_INK, textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel(args.x, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Dice", color=SECONDARY_INK, fontsize=10)
    ax.set_title(f"{args.stage} progress: Dice vs. {args.x} ({len(stage_df)} runs so far)", color=INK, fontsize=12)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{args.stage}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {out_path} ({len(stage_df)} points).")


if __name__ == "__main__":
    main()
