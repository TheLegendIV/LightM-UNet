"""S19 transfer-training strategy comparison (stage 23) -- a dedicated bar
chart, separate from the main Pareto figures, since all four bars here
share the EXACT same architecture (S19's own: dense_dilation_reg_
interleaved_double_mid, separable_dilated=1) and therefore the exact same
cost (params/MACs/mem) -- the only thing varying is the training recipe,
so dice is the only axis worth plotting.

Bars:
  S19 (cold start)  -- the original baseline: 150 epochs on Dataset509 4c,
                        nonneg_block, no pretraining at all.
  23.1 warmstart-4c -- S19's own checkpoint, -pretrained_weights into a
                        fresh 150 more epochs on the SAME 4c problem.
  23.2 binary->4c   -- 150 epochs cold-start on binary Dataset501
                        (nonneg_block throughout), then 150 more on 4c.
  23.3 binary(PReLU)->4c(nonneg) -- same as 23.2 but Phase A uses standard
                        per-channel PReLU, Phase B switches to nonneg_block
                        (slope-initialized from Phase A's own means).

Only stage-23 rows that actually exist in results.csv are plotted --
strategies still mid-training on the cluster are simply absent, not shown
as zero.

Usage:
    python compression/plot_transfer_strategies.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

INK, SECONDARY_INK, MUTED, GRID, SPINE, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
BASELINE_COLOR = "#2a78d6"
STRATEGY_COLOR = "#1baf7a"

# (config_name, display label) -- baseline first, then the 3 strategies in
# the order they were proposed. Missing config_names are skipped, not
# plotted as zero.
BARS: list[tuple[str, str]] = [
    ("nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid", "S19\n(cold start)"),
    ("nnUNetTrainerENet_23_1_s19_warmstart_4c", "23.1\nwarmstart-4c"),
    ("nnUNetTrainerENet_23_2_s19_binary_then_4c", "23.2\nbinary→4c"),
    ("nnUNetTrainerENet_23_3_s19_binary_prelu_then_4c_nonneg", "23.3\nbinary(PReLU)→4c(nonneg)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1
    df = pd.read_csv(args.results_csv)

    rows = []
    for config_name, label in BARS:
        match = df[df["config_name"] == config_name]
        if match.empty:
            print(f"[skip] {config_name} -- not in results.csv yet.")
            continue
        rows.append((label, match.iloc[0]["dice"], config_name == BARS[0][0]))
    if not rows:
        print("None of the stage-23 configs (or the S19 baseline) are in results.csv yet -- nothing to plot.")
        return 1

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    labels = [r[0] for r in rows]
    dices = [r[1] for r in rows]
    colors = [BASELINE_COLOR if r[2] else STRATEGY_COLOR for r in rows]
    x = range(len(rows))
    bars = ax.bar(x, dices, color=colors, zorder=3, width=0.6, edgecolor="black", linewidth=0.5)
    for rect, dice in zip(bars, dices):
        ax.annotate(f"{dice:.4f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=INK)

    baseline_dice = rows[0][1] if rows[0][2] else None
    if baseline_dice is not None:
        ax.axhline(baseline_dice, color=BASELINE_COLOR, linewidth=1, linestyle="--", zorder=2, alpha=0.6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylim(bottom=max(0, min(dices) - 0.05))
    ax.set_title("S19 transfer-training strategy comparison (same architecture, same cost)", color=INK, fontsize=12)

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "s19_transfer_strategies.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(rows)}/{len(BARS)} bars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
