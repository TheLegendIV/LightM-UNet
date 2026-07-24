"""Rank every trained config in compression/results.csv on a unified
hardware-savings-vs-accuracy cost function -- extends
generate_hardware_savings_ranking.py's methodology (established and
confirmed with the user earlier this session: MINIMIZE
score = alpha*macs_ratio + beta*memory_ratio - c*dice_ratio, Dice term
SUBTRACTED not added so higher relative Dice pulls the score down/better)
to every REAL trained result now available (pre-pruning 1a/1b/1c, upscale/
graduated configs, stage1 baselines), not the analytical/placeholder grid
that script covered before any of this was trained.

alpha=beta=c=1/3 (~0.33 each, per this session's instruction).
Baseline: nnUNetTrainerENet_Original (stage1) -- the same anchor used
throughout compression/.

Memory proxy: activation_elements uses the same channel-width-independent
per-stage constants verified via forward hooks in generate_cost_tables.py
(stage2/stage3 share one width and are counted once, matching
activation_memory.csv's convention). MACs proxy: results.csv's `flops`
column is proportional to MACs (flops = 2*macs, from utils.count_flops),
so macs_ratio = flops_ratio identically -- no need to recompute MACs.

Usage:
    python compression/rank_results.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "compression" / "results.csv"
OUT_DIR = REPO_ROOT / "compression" / "results"

ALPHA, BETA, C_WEIGHT = 1 / 3, 1 / 3, 1 / 3
ASSUMED_BITS = 8  # cancels in every ratio -- kept for transparency only
BASELINE_CONFIG = "nnUNetTrainerENet_Original"

STAGE_ELEMENTS_PER_CHANNEL = {"f_i": 65536, "f1": 16384, "stage23": 4096, "f4": 16384, "f5": 65536}

STAGE_COLORS = {
    "stage1": "#2a78d6", "1a_seed": "#eb6834", "1b_maxunpool": "#1baf7a",
    "1c_specialop": "#eda100", "upscale_graduate": "#e87ba4",
}

# Display-only renames (results.csv's config_name is untouched). "Original"
# is ENet exactly as specified in the ENet paper -- the uncompressed
# baseline. 1a_seed_O4_s0 is O4 at native ops/no ablation -- the reference
# point the rest of the pruning study is compared against, i.e. the
# "Compressed Baseline".
DISPLAY_NAME_OVERRIDES = {
    "nnUNetTrainerENet_Original": "ENet",
    "nnUNetTrainerENet_1a_seed_O4_s0": "Compressed Baseline",
}


def display_name(config_name: str) -> str:
    if config_name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[config_name]
    return config_name.replace("nnUNetTrainerENet_", "")


def activation_elements(row: pd.Series) -> float:
    return (
        STAGE_ELEMENTS_PER_CHANNEL["f_i"] * row["f_i"]
        + STAGE_ELEMENTS_PER_CHANNEL["f1"] * row["f1"]
        + STAGE_ELEMENTS_PER_CHANNEL["stage23"] * row["f2"]  # f2==f3 by convention, counted once
        + STAGE_ELEMENTS_PER_CHANNEL["f4"] * row["f4"]
        + STAGE_ELEMENTS_PER_CHANNEL["f5"] * row["f5"]
    )


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"{RESULTS_CSV} not found.")
        return 1
    df = pd.read_csv(RESULTS_CSV)
    n_total = len(df)
    df = df.dropna(subset=["dice", "params", "flops"]).copy()
    if len(df) < n_total:
        print(f"Dropped {n_total - len(df)} rows missing dice/params/flops.")

    baseline_rows = df[df["config_name"] == BASELINE_CONFIG]
    if baseline_rows.empty:
        print(f"Baseline {BASELINE_CONFIG} not found in {RESULTS_CSV} -- can't compute ratios.")
        return 1
    baseline = baseline_rows.iloc[0]

    df["activation_elements"] = df.apply(activation_elements, axis=1)
    baseline_act = activation_elements(baseline)
    baseline_memory = (baseline_act + baseline["params"]) * ASSUMED_BITS
    baseline_macs = baseline["flops"]
    baseline_dice = baseline["dice"]

    df["memory_bits"] = (df["activation_elements"] + df["params"]) * ASSUMED_BITS
    df["macs_ratio"] = df["flops"] / baseline_macs
    df["memory_ratio"] = df["memory_bits"] / baseline_memory
    df["dice_ratio"] = df["dice"] / baseline_dice
    df["score"] = ALPHA * df["macs_ratio"] + BETA * df["memory_ratio"] - C_WEIGHT * df["dice_ratio"]

    df = df.sort_values("score", ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "ranking.csv"
    cols = ["config_name", "stage", "params", "flops", "memory_bits", "dice",
            "macs_ratio", "memory_ratio", "dice_ratio", "score"]
    df[cols].to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} configs, alpha={ALPHA:.3f} beta={BETA:.3f} c={C_WEIGHT:.3f})")
    print(df[["config_name", "stage", "macs_ratio", "memory_ratio", "dice_ratio", "score"]].to_string(index=False))

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

    ax.set_xlabel(
        f"score = {ALPHA:.2f}·MACs_ratio + {BETA:.2f}·memory_ratio − {C_WEIGHT:.2f}·Dice_ratio "
        "(lower = more hardware savings vs. Original, weighted against accuracy loss)",
        color=secondary_ink, fontsize=9,
    )
    ax.set_title("Real trained configs: hardware-savings-vs-accuracy ranking (vs. Original)", color=ink, fontsize=12)

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
