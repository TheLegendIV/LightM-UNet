"""Post-hoc structural-pruning sensitivity analysis for S8-ReLU
(nnUNetTrainerENet_8_2_relu, FP32 dice=0.8218) -- reads every
nnUNetTrainerENet_8_2_relu_prune_* row collect_results.py has written
(compression/results.csv) and plots dice against pruned position, grouped
into three line charts:

  1. Individual blocks, one line per stage (stage2, stage3), x-axis walking
     each stage's own 11 positions in order.
  2. Skip pairs (d=2+8, d=4+16 pruned together within one dilation cycle),
     one line per (stage, cycle) -- 4 series.
  3. Consecutive pairs (d=2+4, d=4+8, d=8+16), same 4 series.

Block naming on the x-axis matches ENet.py's actual content at each
position, not the raw slot index: "0" = the reg-bookend RegularBottleneck
(a real 3x3 conv, full-rank, no dilation), "2"/"4"/"8"/"16" = the
DSCNoProjectionBottleneck at that dilation rate. ("d" = a channel-changing
Downsampling/Upsampling bottleneck -- never pruned by this grid, since
apply_block_pruning's own docstring flags those as unsafe to Identity-out;
included here only for completeness of the naming legend.)

This is a companion to compression/post-quantization/ptq.py in spirit
(reusable, re-run any time collect_results.py adds more prune_* rows) but
scoped to plotting only -- the checkpoint-building/pruning itself is
ENet.py's apply_block_pruning, invoked ad hoc (see the pruning grid's own
build script, not committed -- a one-off driver, not a permanent tool).

Usage:
    python compression/plot_pruning_sensitivity.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
BASELINE_DICE = 0.8218291109668183  # nnUNetTrainerENet_8_2_relu's own real trained dice
BASELINE_COLOR = "#c3392b"

STAGE_COLORS = {"stage2": "#2a78d6", "stage3": "#eb6834"}
CYCLE_COLORS = {
    ("stage2", "A"): "#2a78d6", ("stage2", "B"): "#7cb342",
    ("stage3", "A"): "#eb6834", ("stage3", "B"): "#8e44ad",
}

# Position (0-10) -> ENet.py's real content at that slot, per
# DENSE_DILATION_REG_INTERLEAVED_PATTERN (reg,2,4,8,16,reg,2,4,8,16,reg).
POSITION_LABEL = {0: "0", 1: "2", 2: "4", 3: "8", 4: "16",
                   5: "0", 6: "2", 7: "4", 8: "8", 9: "16", 10: "0"}
# Position -> which cycle it belongs to (A = positions 1-4, B = positions 6-9;
# the reg-bookends at 0/5/10 aren't part of either cycle).
POSITION_CYCLE = {1: "A", 2: "A", 3: "A", 4: "A", 6: "B", 7: "B", 8: "B", 9: "B"}

PRUNE_PREFIX = "nnUNetTrainerENet_8_2_relu_prune_"
INDIVIDUAL_RE = re.compile(r"^stage([23])_(\d+)$")
PAIR_RE = re.compile(r"^stage([23])_(\d+)_(\d+)_(consec|skip)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results_pruning.csv",
                         help="Pruning experiment rows live separately from the main sweep's results.csv -- "
                              "see compression/results_pruning.csv (split out for clarity, ~43 ad-hoc rows "
                              "would otherwise clutter the main sweep table).")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)


def _add_baseline(ax, x_range) -> None:
    ax.axhline(BASELINE_DICE, color=BASELINE_COLOR, linewidth=1.2, linestyle="--",
                zorder=2, label=f"unpruned baseline (dice={BASELINE_DICE:.4f})")


def load_pruning_rows(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df = df[df["config_name"].str.startswith(PRUNE_PREFIX, na=False)].copy()
    df["suffix"] = df["config_name"].str[len(PRUNE_PREFIX):]
    return df


def plot_individual(df: pd.DataFrame, out_dir: Path) -> None:
    """One continuous line walking DEPTH order (stage2.0 -> stage2.10 ->
    stage3.0 -> stage3.10, 22 positions back to back), not two lines
    overlaid on a shared 0-10 axis -- this is the real forward-pass order
    (stage3 runs strictly after stage2, no interleaving), so a single
    depth-ordered walk is the meaningful comparison, with a vertical
    marker at the stage2/stage3 boundary and marker color still carrying
    stage identity."""
    import matplotlib.pyplot as plt

    records = []
    for _, row in df.iterrows():
        m = INDIVIDUAL_RE.match(row["suffix"])
        if not m:
            continue
        stage, pos = f"stage{m.group(1)}", int(m.group(2))
        depth = pos if stage == "stage2" else 11 + pos
        records.append((stage, pos, depth, row["dice"]))
    if not records:
        print("No individual-block pruning rows found yet -- skipping.")
        return
    data = pd.DataFrame(records, columns=["stage", "position", "depth", "dice"]).sort_values("depth")

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    _style_axes(ax)
    _add_baseline(ax, None)
    # One continuous connecting line across the full depth-ordered walk
    # (neutral color -- stage identity is carried by the marker color
    # below, not the line itself).
    ax.plot(data["depth"], data["dice"], color=MUTED, linewidth=1.5, zorder=2)
    for stage, color in STAGE_COLORS.items():
        subset = data[data["stage"] == stage]
        if subset.empty:
            continue
        ax.scatter(subset["depth"], subset["dice"], color=color, s=70, zorder=3,
                   edgecolors="black", linewidths=0.5, label=stage)
    if (data["stage"] == "stage2").any() and (data["stage"] == "stage3").any():
        ax.axvline(10.5, color=GRID, linewidth=1.5, zorder=1)
        ax.annotate("stage2 -> stage3", (10.5, ax.get_ylim()[0]), color=MUTED, fontsize=8,
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    all_depths = sorted(set(data["depth"]))
    ax.set_xticks(all_depths)
    # Two-line ticks: the actual linear depth index on top (what makes this
    # axis genuinely linear-in-depth, not just categorical position-in-
    # stage), content-type code underneath.
    ax.set_xticklabels([f"{d}\n{POSITION_LABEL[d if d <= 10 else d - 11]}" for d in all_depths], fontsize=8)
    ax.set_xlabel("Pruned block depth (0-21, stage2.0..10 then stage3.0..10) -- "
                  "top=depth index, bottom=content (0=reg 3x3, 2/4/8/16=dilation rate)",
                  color=SECONDARY_INK, fontsize=10)
    ax.set_title("S8-ReLU sensitivity: single-block pruning, by depth", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="best")
    fig.tight_layout()
    out_path = out_dir / "pruning_sensitivity_individual.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(data)} points)")


def _depth(stage: str, pos: int) -> int:
    return pos if stage == "stage2" else 11 + pos


def _plot_pairs(df: pd.DataFrame, kind: str, out_name: str, title: str, out_dir: Path) -> None:
    """x-axis is the SAME real depth index (0-21) plot_individual uses --
    anchored at the pair's lower-position block -- not an artificial
    shared category that aligns e.g. stage2's "2+4" with stage3's "2+4"
    at the same x regardless of how far apart they actually sit in the
    network. Each (stage, cycle) line therefore occupies its own true
    depth range (stage2 cycle A ~1-4, stage2 cycle B ~6-9, stage3 cycle A
    ~12-15, stage3 cycle B ~17-20), exactly like the individual-block plot."""
    import matplotlib.pyplot as plt

    records = []
    for _, row in df.iterrows():
        m = PAIR_RE.match(row["suffix"])
        if not m or m.group(4) != kind:
            continue
        stage, a, b = f"stage{m.group(1)}", int(m.group(2)), int(m.group(3))
        cycle = POSITION_CYCLE.get(a)
        if cycle is None or POSITION_CYCLE.get(b) != cycle:
            continue
        depth = _depth(stage, min(a, b))
        content_label = f"{POSITION_LABEL[a]}+{POSITION_LABEL[b]}"
        records.append((stage, cycle, depth, content_label, row["dice"]))
    if not records:
        print(f"No {kind}-pair pruning rows found yet -- skipping.")
        return
    data = pd.DataFrame(records, columns=["stage", "cycle", "depth", "content", "dice"])

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
    _style_axes(ax)
    _add_baseline(ax, None)
    for (stage, cycle), color in CYCLE_COLORS.items():
        subset = data[(data["stage"] == stage) & (data["cycle"] == cycle)].sort_values("depth")
        if subset.empty:
            continue
        ax.plot(subset["depth"], subset["dice"], color=color, linewidth=2, marker="o",
                 markersize=7, markeredgecolor="black", markeredgewidth=0.5, zorder=3,
                 label=f"{stage} cycle {cycle}")
    all_rows = data.sort_values("depth")[["depth", "content"]].drop_duplicates()
    ax.set_xticks(all_rows["depth"])
    ax.set_xticklabels([f"{d}\n{c}" for d, c in zip(all_rows["depth"], all_rows["content"])], fontsize=8)
    ax.set_xlabel("Pruned pair's depth (0-21, anchored at its lower-position block) -- "
                  "top=depth index, bottom=dilation rates pruned", color=SECONDARY_INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="best")
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(data)} points)")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1
    df = load_pruning_rows(args.results_csv)
    if df.empty:
        print(f"No {PRUNE_PREFIX}* rows in {args.results_csv} yet.")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_individual(df, args.out_dir)
    _plot_pairs(df, "consec",
                "pruning_sensitivity_consecutive_pairs.png",
                "S8-ReLU sensitivity: consecutive dilation-pair pruning", args.out_dir)
    _plot_pairs(df, "skip",
                "pruning_sensitivity_skip_pairs.png",
                "S8-ReLU sensitivity: skip dilation-pair pruning", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
