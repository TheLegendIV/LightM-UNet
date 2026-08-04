"""Rank every trained config in compression/results.csv on a unified
cost-vs-accuracy score, for the 4-class objective (LAD/RCA/LCX/LM on
Dataset509_ARCADE_1x1_4c).

score = alpha*macs_ratio + beta*params_ratio - gamma*dice_ratio, alpha=beta=
gamma=1/3 (equal weighting) when a row has a real Dice value; alpha=beta=1/2
(no Dice term at all, not just weight=0) when it doesn't yet -- e.g. a row
collected before training finished, or an architecture-only cost-table entry
with no trained checkpoint behind it. Dice term is SUBTRACTED, not added, so
higher relative Dice pulls the score down/better (minimize = best).

Simplified from the binary-run version of this script (still available in
git history / compression/slurm/archive's era): activation memory is DROPPED
as a factor entirely, per this session's instruction -- only parameter
count, MACs, and Dice feed the score now. (Activation-memory's own table
still exists at generate_cost_tables.py's discretion for anyone who wants
it separately; it's just no longer part of this ranking.)

MACs proxy: results.csv's `flops` column is proportional to MACs (flops =
2*macs, from utils.count_flops), so macs_ratio = flops_ratio identically --
no need to recompute MACs.

Baseline: nnUNetTrainerENet_1_naive_baseline_Baseline (stage
1_naive_baseline's full-width config, ENet-paper channels 16,64,128,64,16 --
the same anchor stage_1's own grid and every downstream stage's probes are
built off of).

Usage:
    python compression/rank_results.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "compression" / "results.csv"
OUT_DIR = REPO_ROOT / "compression" / "results"

EQUAL_WEIGHT_3 = 1 / 3  # macs, params, dice -- when dice is available
EQUAL_WEIGHT_2 = 1 / 2  # macs, params only -- when dice is not (yet) available
BASELINE_CONFIG = "nnUNetTrainerENet_1_naive_baseline_Baseline"

STAGE_COLORS = {
    "1_naive_baseline": "#2a78d6", "2_special_ops": "#eb6834",
    "3_transfer_original": "#1baf7a", "4_arch_probes": "#eda100",
}

# Display-only renames (results.csv's config_name is untouched).
DISPLAY_NAME_OVERRIDES = {
    "nnUNetTrainerENet_1_naive_baseline_Baseline": "Baseline",
    "nnUNetTrainerENet_1_naive_baseline_U4": "U4 (compressed baseline)",
}


def display_name(config_name: str) -> str:
    if config_name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[config_name]
    return config_name.replace("nnUNetTrainerENet_", "")


def row_score(macs_ratio: float, params_ratio: float, dice_ratio: float | None) -> float:
    """dice_ratio=None (no real Dice for this row yet) drops the Dice term
    from the formula entirely and reweights macs/params to 1/2 each, rather
    than leaving it at 1/3 with an implicit zero contribution -- a genuinely
    different (and correct) formula for that row, not a placeholder value
    standing in for Dice."""
    if dice_ratio is None:
        return EQUAL_WEIGHT_2 * macs_ratio + EQUAL_WEIGHT_2 * params_ratio
    return EQUAL_WEIGHT_3 * macs_ratio + EQUAL_WEIGHT_3 * params_ratio - EQUAL_WEIGHT_3 * dice_ratio


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"{RESULTS_CSV} not found.")
        return 1
    df = pd.read_csv(RESULTS_CSV)
    n_total = len(df)
    df = df.dropna(subset=["params", "flops"]).copy()
    if len(df) < n_total:
        print(f"Dropped {n_total - len(df)} rows missing params/flops (need at least those to score at all).")
    if df.empty:
        print("No rows with params/flops -- nothing to rank.")
        return 1

    baseline_rows = df[df["config_name"] == BASELINE_CONFIG]
    if baseline_rows.empty:
        print(f"Baseline {BASELINE_CONFIG} not found in {RESULTS_CSV} -- can't compute ratios.")
        return 1
    baseline = baseline_rows.iloc[0]
    baseline_macs = baseline["flops"]
    baseline_params = baseline["params"]
    baseline_dice = baseline["dice"] if pd.notna(baseline["dice"]) else None
    if baseline_dice is None:
        print(f"WARNING: baseline {BASELINE_CONFIG} itself has no Dice yet -- every row's dice_ratio "
              "will be treated as unavailable (no baseline to ratio against) until it does.")

    df["macs_ratio"] = df["flops"] / baseline_macs
    df["params_ratio"] = df["params"] / baseline_params
    has_dice = df["dice"].notna() & (baseline_dice is not None)
    df["dice_ratio"] = pd.NA
    df.loc[has_dice, "dice_ratio"] = df.loc[has_dice, "dice"] / baseline_dice
    df["dice_is_placeholder"] = ~has_dice

    df["score"] = [
        row_score(row.macs_ratio, row.params_ratio, None if pd.isna(row.dice_ratio) else row.dice_ratio)
        for row in df.itertuples()
    ]

    df = df.sort_values("score", ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "ranking.csv"
    cols = ["config_name", "stage", "params", "flops", "dice", "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM",
            "macs_ratio", "params_ratio", "dice_ratio", "dice_is_placeholder", "score"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} configs, equal-weighted macs/params/dice -- "
          f"{EQUAL_WEIGHT_3:.3f} each when dice present, {EQUAL_WEIGHT_2:.3f} macs/params when not)")
    print(df[["config_name", "stage", "macs_ratio", "params_ratio", "dice_ratio", "dice_is_placeholder", "score"]]
          .to_string(index=False))
    if df["dice_is_placeholder"].any():
        print(f"\nNOTE: {df['dice_is_placeholder'].sum()}/{len(df)} rows have no Dice yet -- scored on "
              "macs_ratio/params_ratio 50/50 only. Re-run after those configs finish training/collecting.")

    plot_ranking(df)
    return 0


def plot_ranking(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, secondary_ink, muted, grid, baseline_color, surface = (
        "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
    )
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(df))), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(True, axis="x", color=grid, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(baseline_color)
    ax.tick_params(colors=muted, labelsize=8)

    labels = [display_name(row.config_name) for row in df.itertuples()]
    colors = [STAGE_COLORS.get(row.stage, "#666666") for row in df.itertuples()]
    y_pos = range(len(df))
    ax.barh(y_pos, df["score"], color=colors, zorder=3, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # best (lowest score) at top
    ax.axvline(0, color=baseline_color, linewidth=1)

    placeholder_note = " (dashed label = no Dice yet, macs/params 50/50 only)" if df["dice_is_placeholder"].any() else ""
    for y, is_placeholder in zip(y_pos, df["dice_is_placeholder"]):
        if is_placeholder:
            ax.get_yticklabels()[y].set_fontstyle("italic")

    ax.set_xlabel(
        f"score = {EQUAL_WEIGHT_3:.2f}·MACs_ratio + {EQUAL_WEIGHT_3:.2f}·params_ratio − {EQUAL_WEIGHT_3:.2f}·Dice_ratio "
        f"(50/50 macs/params if no Dice yet) -- lower = cheaper vs. Baseline, weighted against accuracy loss"
        + placeholder_note,
        color=secondary_ink, fontsize=9,
    )
    ax.set_title("4-class objective: cost-vs-accuracy ranking (vs. Baseline)", color=ink, fontsize=12)

    present_stages = [s for s in STAGE_COLORS if s in df["stage"].unique()]
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s) for s in present_stages]
    ax.legend(handles=handles, title="stage", frameon=False, fontsize=8, labelcolor=secondary_ink, loc="lower right")

    fig.tight_layout()
    out_path = OUT_DIR / "ranking.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
