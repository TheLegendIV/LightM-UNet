"""Post-hoc structural-pruning sensitivity analysis -- reads every
<prefix>_prune_* row collect_results.py has written
(compression/results_pruning.csv) and plots dice against pruned position,
grouped into three line charts per model:

  1. Individual blocks, one line per stage (stage2, stage3), x-axis walking
     each stage's own positions in order.
  2. Consecutive pairs (i, i+1) -- EVERY adjacent pair within a stage, not
     just ones sitting inside a dilation cycle (2+4, 4+8, 8+16): also
     covers pairs touching a reg-bookend slot (e.g. 0+2, 16+0) and pairs
     straddling the two dilation cycles.
  3. Skip pairs (i, i+2) -- same full-range coverage, one gap apart.

One line per stage (stage2, stage3), matching plot_individual's own
depth-ordered-walk style -- NOT grouped by dilation "cycle" (that grouping
only made sense when pairs were restricted to within-cycle positions; once
every pair is included, reg-bookend-involving pairs don't belong to either
cycle).

Supports multiple source models via MODEL_CONFIGS below (currently
S8-ReLU's 11-slot dense_dilation_reg_interleaved pattern and S19's 12-slot
dense_dilation_reg_interleaved_double_mid pattern -- the doubled mid-cycle
reg bookend adds a second bookend slot at 5/6). Each model gets its own
three PNGs (S19's carry an "_s19" suffix; S8-ReLU's keep their original
unsuffixed names for backward compatibility) -- deliberately NOT overlaid
on a shared axis, since the two patterns have different slot counts and
the same x-position would mean different content between them. S8-ReLU's
own results_pruning.csv rows still only cover the original within-cycle
pairs (not backfilled to all-pairs) -- its plots will just show fewer
points in the same style, no code changes needed either way.

Block naming on the x-axis matches ENet.py's actual content at each
position, not the raw slot index: "0" = the reg-bookend RegularBottleneck
(a real 3x3 conv, full-rank, no dilation), "2"/"4"/"8"/"16" = the dilated
bottleneck at that rate. ("d" = a channel-changing Downsampling/Upsampling
bottleneck -- never pruned by this grid, since apply_block_pruning's own
docstring flags those as unsafe to Identity-out; included here only for
completeness of the naming legend.)

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
BASELINE_COLOR = "#c3392b"

STAGE_COLORS = {"stage2": "#2a78d6", "stage3": "#eb6834"}

INDIVIDUAL_RE = re.compile(r"^stage([23])_(\d+)$")
PAIR_RE = re.compile(r"^stage([23])_(\d+)_(\d+)_(consec|skip)$")

# Per-model config: prefix (results_pruning.csv's config_name prefix to
# filter on), the model's own real trained FP32 dice (unpruned baseline,
# for the dashed reference line), how many slots its context pattern has
# per stage, and the position->content-label map (keyed by raw slot index,
# matching ENet.py's own pattern definition -- see
# DENSE_DILATION_REG_INTERLEAVED_PATTERN /
# DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN).
MODEL_CONFIGS = {
    "s8relu": {
        "prefix": "nnUNetTrainerENet_8_2_relu_prune_",
        "baseline_dice": 0.8218291109668183,
        "n_slots": 11,
        "position_label": {0: "0", 1: "2", 2: "4", 3: "8", 4: "16",
                            5: "0", 6: "2", 7: "4", 8: "8", 9: "16", 10: "0"},
        "title_prefix": "S8-ReLU",
        "out_suffix": "",
    },
    "s19": {
        "prefix": "nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid_prune_",
        "baseline_dice": 0.793139270492374,
        "n_slots": 12,
        "position_label": {0: "0", 1: "2", 2: "4", 3: "8", 4: "16",
                            5: "0", 6: "0", 7: "2", 8: "4", 9: "8", 10: "16", 11: "0"},
        "title_prefix": "S19",
        "out_suffix": "_s19",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results_pruning.csv",
                         help="Pruning experiment rows live separately from the main sweep's results.csv -- "
                              "see compression/results_pruning.csv (split out for clarity).")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS), default=list(MODEL_CONFIGS),
                         help="Which model(s) in MODEL_CONFIGS to plot (default: all).")
    return parser.parse_args()


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)


def _add_baseline(ax, baseline_dice: float) -> None:
    ax.axhline(baseline_dice, color=BASELINE_COLOR, linewidth=1.2, linestyle="--",
                zorder=2, label=f"unpruned baseline (dice={baseline_dice:.4f})")


def load_pruning_rows(results_csv: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df = df[df["config_name"].str.startswith(prefix, na=False)].copy()
    df["suffix"] = df["config_name"].str[len(prefix):]
    return df


def plot_individual(df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """One continuous line walking DEPTH order (stage2.0 -> stage2.(n-1) ->
    stage3.0 -> stage3.(n-1), 2*n_slots positions back to back), not two
    lines overlaid on a shared 0..(n-1) axis -- this is the real forward-
    pass order (stage3 runs strictly after stage2, no interleaving), so a
    single depth-ordered walk is the meaningful comparison, with a vertical
    marker at the stage2/stage3 boundary and marker color still carrying
    stage identity."""
    import matplotlib.pyplot as plt

    n_slots = cfg["n_slots"]
    position_label = cfg["position_label"]
    records = []
    for _, row in df.iterrows():
        m = INDIVIDUAL_RE.match(row["suffix"])
        if not m:
            continue
        stage, pos = f"stage{m.group(1)}", int(m.group(2))
        depth = pos if stage == "stage2" else n_slots + pos
        records.append((stage, pos, depth, row["dice"]))
    if not records:
        print(f"[{cfg['title_prefix']}] No individual-block pruning rows found yet -- skipping.")
        return
    data = pd.DataFrame(records, columns=["stage", "position", "depth", "dice"]).sort_values("depth")

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    _style_axes(ax)
    _add_baseline(ax, cfg["baseline_dice"])
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
        boundary = n_slots - 0.5
        ax.axvline(boundary, color=GRID, linewidth=1.5, zorder=1)
        ax.annotate("stage2 -> stage3", (boundary, ax.get_ylim()[0]), color=MUTED, fontsize=8,
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    all_depths = sorted(set(data["depth"]))
    ax.set_xticks(all_depths)
    # Two-line ticks: the actual linear depth index on top (what makes this
    # axis genuinely linear-in-depth, not just categorical position-in-
    # stage), content-type code underneath.
    ax.set_xticklabels([f"{d}\n{position_label[d if d < n_slots else d - n_slots]}" for d in all_depths], fontsize=8)
    ax.set_xlabel(f"Pruned block depth (0-{2 * n_slots - 1}, stage2.0..{n_slots - 1} then stage3.0..{n_slots - 1}) -- "
                  "top=depth index, bottom=content (0=reg 3x3, 2/4/8/16=dilation rate)",
                  color=SECONDARY_INK, fontsize=10)
    ax.set_title(f"{cfg['title_prefix']} sensitivity: single-block pruning, by depth", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="best")
    fig.tight_layout()
    out_path = out_dir / f"pruning_sensitivity_individual{cfg['out_suffix']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(data)} points)")


def _depth(stage: str, pos: int, n_slots: int) -> int:
    return pos if stage == "stage2" else n_slots + pos


def _plot_pairs(df: pd.DataFrame, cfg: dict, kind: str, out_name: str, title: str, out_dir: Path) -> None:
    """x-axis is the SAME real depth index plot_individual uses -- anchored
    at the pair's lower-position block -- not an artificial shared category
    that aligns e.g. stage2's "2+4" with stage3's "2+4" at the same x
    regardless of how far apart they actually sit in the network. One line
    per STAGE (not per dilation cycle -- covers every within-stage pair,
    including ones touching a reg-bookend slot or straddling both dilation
    cycles, so "cycle" no longer cleanly partitions the data), matching
    plot_individual's own depth-ordered-walk style."""
    import matplotlib.pyplot as plt

    n_slots = cfg["n_slots"]
    position_label = cfg["position_label"]
    records = []
    for _, row in df.iterrows():
        m = PAIR_RE.match(row["suffix"])
        if not m or m.group(4) != kind:
            continue
        stage, a, b = f"stage{m.group(1)}", int(m.group(2)), int(m.group(3))
        depth = _depth(stage, min(a, b), n_slots)
        content_label = f"{position_label[a]}+{position_label[b]}"
        records.append((stage, depth, content_label, row["dice"]))
    if not records:
        print(f"[{cfg['title_prefix']}] No {kind}-pair pruning rows found yet -- skipping.")
        return
    data = pd.DataFrame(records, columns=["stage", "depth", "content", "dice"])

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    _style_axes(ax)
    _add_baseline(ax, cfg["baseline_dice"])
    for stage, color in STAGE_COLORS.items():
        subset = data[data["stage"] == stage].sort_values("depth")
        if subset.empty:
            continue
        ax.plot(subset["depth"], subset["dice"], color=color, linewidth=2, marker="o",
                 markersize=7, markeredgecolor="black", markeredgewidth=0.5, zorder=3,
                 label=stage)
    if (data["stage"] == "stage2").any() and (data["stage"] == "stage3").any():
        boundary = n_slots - 0.5
        ax.axvline(boundary, color=GRID, linewidth=1.5, zorder=1)
        ax.annotate("stage2 -> stage3", (boundary, ax.get_ylim()[0]), color=MUTED, fontsize=8,
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    all_rows = data.sort_values("depth")[["depth", "content"]].drop_duplicates()
    ax.set_xticks(all_rows["depth"])
    ax.set_xticklabels([f"{d}\n{c}" for d, c in zip(all_rows["depth"], all_rows["content"])], fontsize=8)
    ax.set_xlabel("Pruned pair's depth (anchored at its lower-position block) -- "
                  "top=depth index, bottom=content codes pruned (0=reg 3x3, 2/4/8/16=dilation rate)",
                  color=SECONDARY_INK, fontsize=10)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    any_written = False
    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        df = load_pruning_rows(args.results_csv, cfg["prefix"])
        if df.empty:
            print(f"No {cfg['prefix']}* rows in {args.results_csv} yet.")
            continue
        any_written = True
        plot_individual(df, cfg, args.out_dir)
        _plot_pairs(df, cfg, "consec",
                    f"pruning_sensitivity_consecutive_pairs{cfg['out_suffix']}.png",
                    f"{cfg['title_prefix']} sensitivity: consecutive-pair pruning (all pairs)", args.out_dir)
        _plot_pairs(df, cfg, "skip",
                    f"pruning_sensitivity_skip_pairs{cfg['out_suffix']}.png",
                    f"{cfg['title_prefix']} sensitivity: skip-pair pruning (all pairs)", args.out_dir)
    return 0 if any_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
