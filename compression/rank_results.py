"""Rank every trained config in compression/results.csv on a unified
cost-vs-accuracy score, for the 4-class objective (LAD/RCA/LCX/LM on
Dataset509_ARCADE_1x1_4c).

score = 0.25*macs_ratio + 0.25*params_ratio + 0.25*mem_ratio - 0.25*dice_ratio,
each ratio taken against U4 (nnUNetTrainerENet_1_naive_baseline_U4-- the
compressed reference every downstream architecture probe is actually built
off of, not the full-width Baseline). Equal 0.25 weighting across all four
metrics when a row has real values for all of them; any metric missing for
a given row (no Dice yet, no mem_elements backfilled, etc.) drops OUT of
the formula entirely for that row and the remaining metrics are reweighted
to split 1.0 equally among themselves -- a genuinely different (and
correct) formula for that row, not a placeholder value standing in. Dice
term is SUBTRACTED, not added, so higher relative Dice pulls the score
down/better (minimize = best) -- consistent with the cost terms, where
lower/cheaper is also better.

MACs proxy: results.csv's `flops` column is proportional to MACs (flops =
2*macs, from utils.count_flops), so macs_ratio = flops_ratio identically --
no need to recompute MACs.

Baseline: nnUNetTrainerENet_1_naive_baseline_U4 (stage 1_naive_baseline's
own U4-width config, channels=4,16,32,16,4 -- the actual shared reference
point every arch-probe stage from stage_4 onward is built off of; see
plot_arch_probes_focused.py's own U4-anchored framing).

Usage:
    python compression/rank_results.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "compression" / "results.csv"
ABBREV_CSV = REPO_ROOT / "compression" / "config_abbreviations.csv"
OUT_DIR = REPO_ROOT / "compression" / "results"

METRIC_COLS = ("macs_ratio", "params_ratio", "mem_ratio", "dice_ratio")
BASELINE_CONFIG = "nnUNetTrainerENet_1_naive_baseline_U4"

STAGE_COLORS = {
    "1_naive_baseline": "#2a78d6", "2_special_ops": "#eb6834",
    "3_transfer_original": "#1baf7a", "4_arch_probes": "#eda100",
}

# Display-only renames (results.csv's config_name is untouched).
DISPLAY_NAME_OVERRIDES = {
    "nnUNetTrainerENet_1_naive_baseline_Baseline": "Baseline (full width)",
    "nnUNetTrainerENet_1_naive_baseline_U4": "U4 (reference)",
}


def load_abbreviations() -> dict[str, str]:
    if not ABBREV_CSV.exists():
        return {}
    abbrev_df = pd.read_csv(ABBREV_CSV)
    return dict(zip(abbrev_df["config_name"], abbrev_df["abbrev"]))


def display_name(config_name: str, abbrev: dict[str, str]) -> str:
    if config_name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[config_name]
    if config_name in abbrev:
        return abbrev[config_name]
    return config_name.replace("nnUNetTrainerENet_", "")


def row_score(ratios: dict[str, float | None]) -> tuple[float, int]:
    """Any ratio that's None (metric unavailable for this row) drops out of
    the formula entirely -- the remaining ones split weight 1.0 equally
    among themselves, not 0.25 each with an implicit zero. dice_ratio is
    SUBTRACTED (higher relative dice = lower/better score); the three cost
    ratios are ADDED (lower/cheaper = lower/better score). Returns
    (score, n_metrics_used) so callers can flag partially-scored rows."""
    present = {k: v for k, v in ratios.items() if v is not None}
    if not present:
        return float("nan"), 0
    weight = 1.0 / len(present)
    score = sum((-v if k == "dice_ratio" else v) * weight for k, v in present.items())
    return score, len(present)


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
    baseline_mem = baseline["mem_elements"] if "mem_elements" in baseline and pd.notna(baseline["mem_elements"]) else None
    baseline_dice = baseline["dice"] if pd.notna(baseline["dice"]) else None
    if baseline_dice is None:
        print(f"WARNING: baseline {BASELINE_CONFIG} itself has no Dice yet -- every row's dice_ratio "
              "will be treated as unavailable (no baseline to ratio against) until it does.")
    if baseline_mem is None:
        print(f"WARNING: baseline {BASELINE_CONFIG} itself has no mem_elements yet -- every row's mem_ratio "
              "will be treated as unavailable until it does.")

    df["macs_ratio"] = df["flops"] / baseline_macs
    df["params_ratio"] = df["params"] / baseline_params
    df["mem_ratio"] = pd.NA
    if baseline_mem is not None and "mem_elements" in df.columns:
        has_mem = df["mem_elements"].notna()
        df.loc[has_mem, "mem_ratio"] = df.loc[has_mem, "mem_elements"] / baseline_mem
    df["dice_ratio"] = pd.NA
    if baseline_dice is not None:
        has_dice = df["dice"].notna()
        df.loc[has_dice, "dice_ratio"] = df.loc[has_dice, "dice"] / baseline_dice

    scores = [
        row_score({
            "macs_ratio": row.macs_ratio,
            "params_ratio": row.params_ratio,
            "mem_ratio": None if pd.isna(row.mem_ratio) else row.mem_ratio,
            "dice_ratio": None if pd.isna(row.dice_ratio) else row.dice_ratio,
        })
        for row in df.itertuples()
    ]
    df["score"] = [s for s, _ in scores]
    df["n_metrics_used"] = [n for _, n in scores]
    df["score_is_partial"] = df["n_metrics_used"] < 4

    df = df.sort_values("score", ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "ranking.csv"
    cols = ["config_name", "stage", "params", "flops", "mem_elements", "dice",
            "dice_LAD", "dice_RCA", "dice_LCX", "dice_LM",
            "macs_ratio", "params_ratio", "mem_ratio", "dice_ratio",
            "n_metrics_used", "score_is_partial", "score"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} configs, 0.25 each of macs/params/mem/dice ratio vs. {BASELINE_CONFIG} "
          f"when all 4 are present; missing metrics drop out and the rest reweight equally per-row)")
    print(df[["config_name", "stage", "macs_ratio", "params_ratio", "mem_ratio", "dice_ratio",
              "n_metrics_used", "score"]].to_string(index=False))
    if df["score_is_partial"].any():
        print(f"\nNOTE: {df['score_is_partial'].sum()}/{len(df)} rows scored on fewer than all 4 metrics "
              "(missing dice and/or mem_elements) -- see n_metrics_used per row.")

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

    abbrev = load_abbreviations()
    labels = [display_name(row.config_name, abbrev) for row in df.itertuples()]
    colors = [STAGE_COLORS.get(row.stage, "#666666") for row in df.itertuples()]
    y_pos = range(len(df))
    ax.barh(y_pos, df["score"], color=colors, zorder=3, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # best (lowest score) at top
    ax.axvline(0, color=baseline_color, linewidth=1)

    placeholder_note = " (italic label = fewer than 4 metrics available, see n_metrics_used)" if df["score_is_partial"].any() else ""
    for y, is_partial in zip(y_pos, df["score_is_partial"]):
        if is_partial:
            ax.get_yticklabels()[y].set_fontstyle("italic")

    ax.set_xlabel(
        "score = 0.25·MACs_ratio + 0.25·params_ratio + 0.25·mem_ratio − 0.25·Dice_ratio, each vs. U4 "
        "(equal reweight among available metrics if any are missing for a row) -- "
        "lower = cheaper/less-memory vs. U4, weighted against accuracy loss"
        + placeholder_note,
        color=secondary_ink, fontsize=9,
    )
    ax.set_title("4-class objective: cost-vs-accuracy ranking (vs. U4)", color=ink, fontsize=12)

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
