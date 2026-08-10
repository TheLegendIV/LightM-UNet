"""Final proof-of-beat plot: stage_1_naive_baseline's own channel-width
curve (Base/U2/U4/U8/U16) vs. the TRUE Pareto front across every named
config in compression/config_abbreviations.csv -- for all three cost
metrics (params, MACs, FINN buffer memory elements). Answers "what's the
best dice actually achievable at any given cost, across the whole sweep,
and does it beat the naive width-compression curve" -- not just one
architecture's own width sweep (see plot_all_configs.py for the
everything-at-once view this is a focused derivative of).

Pareto front = the standard efficiency-frontier definition: sort by cost
ascending, keep a point only if its dice strictly exceeds every
cheaper-or-equal point's dice seen so far (a "staircase" of non-dominated
points). Computed separately per metric (a config Pareto-optimal in params
need not be Pareto-optimal in MACs or memory elements too), then the SAME
markers -- the UNION of all three per-metric fronts, i.e. every config
that's Pareto-optimal on params OR macs OR mem_elements -- are drawn in all
three figures. A point that only wins on one axis still shows up
(off-frontier on the other two), so the same shortlist of contenders can be
visually tracked across all three figures instead of a different, disjoint
set of markers silently appearing/disappearing per figure.

The naive baseline curve is drawn as plain straight-line (piecewise-linear)
segments, not a smoothing spline -- see plot_all_configs.py's own module
comment for why (sparse real points, no real underlying smooth function,
straight lines are the standard convention for this kind of tradeoff
figure). The Pareto front is markers only, deliberately NOT connected by a
line -- unlike the naive curve, its points come from entirely different
architectures with no shared axis between them (channel width), so a
connecting line would visually imply a continuous tradeoff that doesn't
exist between e.g. S9.4 and S13.1. Front markers are an X. Configs in
HIGHLIGHTED_CONFIGS (currently S19 and S5.3) get their own colored diamond
drawn on top, but ONLY when they're not already part of the (union)
Pareto front -- the whole point of the diamond is to force a look at a
config the front left out; a config that genuinely IS on the front already
gets a normal X like everyone else, not a redundant second marker on top
of itself.

Only rows with a compression/config_abbreviations.csv entry are
considered (excludes pruning-grid rows, which live in results_pruning.csv
entirely, and quantization/experiment rows, which mix in a different axis
-- bit-width -- not directly comparable to an FP32 architecture sweep).

Usage:
    python compression/plot_final_width_comparison.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ABBREV_CSV = Path(__file__).resolve().parent / "config_abbreviations.csv"

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
NAIVE_COLOR = "#2a78d6"
PARETO_COLOR = "#1baf7a"
S19_COLOR = "#d6272a"

NAIVE_STAGE = "1_naive_baseline"

# Configs called out by name elsewhere (the paper's comparison table, or
# just worth pointing at directly) -- each gets its own always-shown
# diamond, drawn on top of every other layer regardless of whether it's
# actually Pareto-optimal on that particular metric, in a color that
# distinguishes it from the rest of the front and from every other
# highlighted config.
S19_COLDSTART_CONFIG = "nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid"
S5_3_CONFIG = "nnUNetTrainerENet_5_3_dense_dilation_merge_pairs"
# value = (marker color, legend label override -- None falls back to the
# config's own config_abbreviations.csv abbrev, e.g. S5.3's "S5.3").
HIGHLIGHTED_CONFIGS = {
    S19_COLDSTART_CONFIG: (S19_COLOR, "S19"),
    S5_3_CONFIG: ("#e6a817", None),
}

# Runs trained via nnU-Net's own -pretrained_weights transfer (warm-started
# from an already-trained checkpoint) rather than from scratch, so their
# dice reflects head-start training, not the architecture alone -- not an
# apples-to-apples comparison against the rest of the from-scratch sweep.
# Excluded from the Pareto front entirely (all three: S3.1 warm-starts from
# the old binary Dataset501 checkpoint; S13.1/S13.2 warm-start from
# 5_6_separable_dense_dilation's checkpoint, S13.2 additionally freezing the
# transferred leaky-slope scalars for part of training).
UNFAITHFUL_TRAINING_CONFIGS = {
    "nnUNetTrainerENet_3_transfer_original",
    "nnUNetTrainerENet_13_separable_dense_nonneg_block_warmstart",
    "nnUNetTrainerENet_13_separable_dense_nonneg_block_leaky_frozen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def load_abbreviations() -> pd.DataFrame:
    return pd.read_csv(ABBREV_CSV)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)
    ax.set_xscale("log")


def pareto_front(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Standard non-dominated-point staircase: sort by cost ascending, keep
    a point only if its dice strictly beats every cheaper-or-equal point's
    dice seen so far."""
    d = df.dropna(subset=[x_col, "dice"]).sort_values(x_col)
    keep = []
    best_dice = -float("inf")
    for _, row in d.iterrows():
        if row["dice"] > best_dice:
            keep.append(row)
            best_dice = row["dice"]
    return pd.DataFrame(keep)


def union_pareto_configs(pareto_eligible: pd.DataFrame, metric_cols: tuple[str, ...]) -> set[str]:
    """Every config that's Pareto-optimal on AT LEAST ONE of metric_cols --
    the union, not the intersection, of the per-metric fronts. This is the
    shortlist drawn (consistently) across all three figures."""
    union: set[str] = set()
    for col in metric_cols:
        union |= set(pareto_front(pareto_eligible, col)["config_name"])
    return union


def _plot_metric(named: pd.DataFrame, naive: pd.DataFrame, pareto_eligible: pd.DataFrame,
                  front_configs: set[str], x_col: str, x_label: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=SURFACE)
    _style_axes(ax)

    naive_sorted = naive.dropna(subset=[x_col, "dice"]).sort_values(x_col)
    front = (pareto_eligible[pareto_eligible["config_name"].isin(front_configs)]
              .dropna(subset=[x_col, "dice"]).sort_values(x_col))
    # A naive point that's ALSO Pareto-optimal (e.g. the naive curve's own
    # cheapest/most-efficient points often are) gets its marker+label drawn
    # once below, from the Pareto loop -- skip it here so it isn't
    # double-plotted/double-labeled on top of itself.
    on_front = set(front["config_name"]) if not front.empty else set()
    naive_off_front = naive_sorted[~naive_sorted["config_name"].isin(on_front)]
    if len(naive_sorted) >= 2:
        ax.plot(naive_sorted[x_col], naive_sorted["dice"], color=NAIVE_COLOR, linewidth=1.5,
                 zorder=2, label="naive width-compression curve")
    ax.scatter(naive_off_front[x_col], naive_off_front["dice"], color=NAIVE_COLOR, s=70, zorder=3,
               edgecolors="black", linewidths=0.5)
    for _, row in naive_off_front.iterrows():
        ax.annotate(row["abbrev"], (row[x_col], row["dice"]), fontsize=7.5, color=SECONDARY_INK,
                   textcoords="offset points", xytext=(5, -10))

    # Highlighted configs (HIGHLIGHTED_CONFIGS) get their own colored
    # diamond, on top of every other layer -- but only for configs that
    # actually NEED forcing, i.e. aren't already in the (union) Pareto
    # front. A highlighted config that's already Pareto-optimal on at
    # least one metric has nothing to force -- it already gets a normal X
    # marker below, and drawing a redundant diamond on top of its own X
    # would just be double-marking the same point.
    diamond_configs = {c: v for c, v in HIGHLIGHTED_CONFIGS.items() if c not in front_configs}
    front_other = front[~front["config_name"].isin(diamond_configs)] if not front.empty else front
    if not front_other.empty:
        # Lowercase "x" -- matplotlib's thin, unfilled line-cross marker
        # (like a text "X" glyph), not the bold uppercase "X" filled marker.
        ax.scatter(front_other[x_col], front_other["dice"], color=PARETO_COLOR, s=90, zorder=5,
                   marker="x", linewidths=1.8, label="Pareto front (union of all 3 metrics)")
        # Alternate the label offset up/down (and vary horizontal reach a
        # little) so densely-packed Pareto points -- common near the
        # "knee" of the curve where several architectures land close
        # together -- don't render with fully overlapping text.
        for i, (_, row) in enumerate(front_other.iterrows()):
            y_off = 9 if i % 2 == 0 else -15
            x_off = 6 + 4 * (i % 3)
            ax.annotate(row["abbrev"], (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                       fontweight="bold", textcoords="offset points", xytext=(x_off, y_off))
    for i, (config_name, (color, legend_label)) in enumerate(diamond_configs.items()):
        highlight_row = named[named["config_name"] == config_name].dropna(subset=[x_col, "dice"])
        if highlight_row.empty:
            continue
        label = legend_label or highlight_row.iloc[0]["abbrev"]
        ax.scatter(highlight_row[x_col], highlight_row["dice"], color=color, s=140, zorder=6,
                   marker="D", edgecolors="black", linewidths=0.9, label=label)
        for _, row in highlight_row.iterrows():
            ax.annotate(row["abbrev"], (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                       fontweight="bold", textcoords="offset points", xytext=(8, 9 - 14 * i))

    ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
    ax.set_title(f"Pareto front vs. naive width-compression curve: Dice vs. {x_label}", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    highlighted_present = [named[named["config_name"] == c]["abbrev"].iloc[0]
                            for c in diamond_configs
                            if not named[named["config_name"] == c].dropna(subset=[x_col, "dice"]).empty]
    highlight_note = f" +{'/'.join(highlighted_present)}" if highlighted_present else ""
    print(f"Wrote {out_path} ({len(front_other)} Pareto-front points{highlight_note})")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1
    df = pd.read_csv(args.results_csv)
    abbrev_df = load_abbreviations()

    named = df.merge(abbrev_df[["abbrev", "config_name"]], on="config_name", how="inner")
    if named.empty:
        print("No results.csv rows matched a config_abbreviations.csv entry.")
        return 1
    named = named.copy()
    named["macs"] = named["flops"] / 2

    naive = named[named["stage"] == NAIVE_STAGE]
    if naive.empty:
        print(f"No {NAIVE_STAGE} rows found.")
        return 1

    pareto_eligible = named[~named["config_name"].isin(UNFAITHFUL_TRAINING_CONFIGS)]
    metric_cols = ("params", "macs", "mem_elements")
    front_configs = union_pareto_configs(pareto_eligible, metric_cols)
    print(f"Pareto front (union of {metric_cols}): {len(front_configs)} configs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(named, naive, pareto_eligible, front_configs, "params", "Parameters",
                 args.out_dir / "dice_vs_params_final.png")
    _plot_metric(named, naive, pareto_eligible, front_configs, "macs", "MACs",
                 args.out_dir / "dice_vs_macs_final.png")
    _plot_metric(named, naive, pareto_eligible, front_configs, "mem_elements",
                 "FINN buffer memory (activation elements)", args.out_dir / "dice_vs_mem_elements_final.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
