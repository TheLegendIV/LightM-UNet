"""Original-structure-preserving reduction family: how much hardware can be
saved by shrinking Original while keeping f_i fixed, vs. jumping to a
different filter family (section 2a's O2/O4/O8/O16/OF grid).

Score (per this session's instruction -- ACCURACY REMOVED ENTIRELY, unlike
hardware_savings_ranking.csv's alpha=beta=c=1/3):
    score = alpha*macs_ratio + beta*memory_ratio,  alpha=beta=0.5
    savings = 1 - score   (0% at Original itself, by construction -- sanity-checked below)

Construction: f_i (=16) fixed. f1, f2=f3, f4, f5 each reduced by 8 per step,
independently clamped at a floor of 4 (matches the grid search's own floor
-- OF is (16,4,4,4,4)) -- once a stage hits 4 it stays there while whichever
stages haven't floored yet keep reducing. f5 floors first (starts at 16,
same magnitude as f_i despite not being tied to it), then f1/f4 (start at
64), then f2/f3 last (start at 128, the widest). The scheme is exhausted
once every stage is at the floor -- which lands exactly on OF (16,4,4,4,4),
a clean internal check that this construction is consistent with the grid.

Re-baselined from E1 to Original this session: real test-set inference
showed Original beats E1 on Dice (0.7800 vs 0.7746) while already being
smaller -- see foundation_log.md's "Plan revision 2" section. The new
reference pick (O4) was found directly via the geometric-axis method (see
generate_hardware_savings_ranking.py) without needing this interpolation --
this script is re-run for completeness/consistency, not because O4's
derivation depended on it.

Earlier E1-baselined version fixed f5 alongside f_i, on the (incorrect)
assumption that the grid search required f_i==f5 symmetry throughout --
checked directly against the actual grid definitions and it doesn't: only
the "full width" baseline (E1, now Original) has that symmetry (inherited
from being the paper-faithful/tuned reference, not a requirement), the
scaled-down configs all break it and are valid under upsample_conv (only
max_unpool requires f_i==f5). This version fixes f_i only, per that
correction.

Usage: python compression/generate_symmetric_reduction_family.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_flops, count_params  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "cost_tables"
ALPHA, BETA = 0.5, 0.5  # accuracy term removed entirely, per this session's instruction
ASSUMED_BITS = 8  # cancels in every ratio, see hardware_savings_ranking.py's docstring

ORIGINAL_CHANNELS = (16, 64, 128, 64, 16)
BOTTLENECKS = (4, 8, 8, 2, 1)  # bnnative, fixed throughout
DECODER_TYPE = "upsample_conv"
INPUT_HW = (512, 512)
STEP = 8
FLOOR = 4  # matches the grid search's own floor (OF)
TARGET_SAVINGS = 0.5

STAGE_ELEMENTS_PER_CHANNEL = {"f_i": 65536, "f1": 16384, "stage23": 4096, "f4": 16384, "f5": 65536}

FILTER_CHANNELS = {
    "Original": (16, 64, 128, 64, 16), "O2": (16, 32, 64, 32, 8), "O4": (16, 16, 32, 16, 4),
    "O8": (16, 8, 16, 8, 4), "O16": (16, 4, 8, 4, 4), "OF": (16, 4, 4, 4, 4),
}


def activation_elements(channels: tuple[int, ...]) -> int:
    f_i, f1, stage23, f4, f5 = channels
    return (STAGE_ELEMENTS_PER_CHANNEL["f_i"] * f_i + STAGE_ELEMENTS_PER_CHANNEL["f1"] * f1
            + STAGE_ELEMENTS_PER_CHANNEL["stage23"] * stage23 + STAGE_ELEMENTS_PER_CHANNEL["f4"] * f4
            + STAGE_ELEMENTS_PER_CHANNEL["f5"] * f5)


def measure(channels: tuple[int, ...]) -> dict:
    model = ENet(in_channels=1, out_channels=2, channels=channels,
                 bottlenecks_per_stage=BOTTLENECKS, decoder_type=DECODER_TYPE)
    params, _ = count_params(model)
    macs, _ = count_flops(model, 1, INPUT_HW)
    act_elems = activation_elements(channels)
    memory_bits = (act_elems + params) * ASSUMED_BITS
    return {"params": params, "macs": macs, "memory_bits": memory_bits, "total_channels": sum(channels)}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    original = measure(ORIGINAL_CHANNELS)
    print(f"Original baseline: {ORIGINAL_CHANNELS} params={original['params']} "
          f"macs={original['macs']:.3e} memory_bits={original['memory_bits']:.3e}")

    family_rows = []
    k = 0
    while True:
        f1 = max(FLOOR, 64 - STEP * k)
        f23 = max(FLOOR, 128 - STEP * k)
        f4 = max(FLOOR, 64 - STEP * k)
        f5 = max(FLOOR, 16 - STEP * k)
        channels = (16, f1, f23, f4, f5)
        m = measure(channels)
        macs_ratio = m["macs"] / original["macs"]
        mem_ratio = m["memory_bits"] / original["memory_bits"]
        savings = 1 - (ALPHA * macs_ratio + BETA * mem_ratio)
        family_rows.append({
            "config_name": "Original" if k == 0 else f"Original_fixfi_k{k}",
            "channels": str(channels), "k": k, "total_channels": m["total_channels"],
            "params": m["params"], "macs": m["macs"], "memory_bits": m["memory_bits"],
            "macs_ratio": macs_ratio, "memory_ratio": mem_ratio, "savings": savings,
        })
        if f1 == FLOOR and f23 == FLOOR and f4 == FLOOR and f5 == FLOOR:
            break  # exhausted -- this is OF
        k += 1
    family_df = pd.DataFrame(family_rows)
    assert abs(family_df.iloc[0]["savings"]) < 1e-9, "Original (k=0) must give exactly 0% savings"
    assert family_df.iloc[-1]["channels"] == str(FILTER_CHANNELS["OF"]), "scheme must converge exactly to OF"
    print(f"Original (k=0) savings = {family_df.iloc[0]['savings']*100:.6f}% -- sanity check passed (must be 0).")
    print(f"Final point = {family_df.iloc[-1]['channels']} -- confirmed identical to OF.")

    out_csv = OUT_DIR / "symmetric_reduction_family.csv"
    family_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(family_df)} points, k=0..{len(family_df)-1})")
    print(family_df[["config_name", "channels", "macs_ratio", "memory_ratio", "savings"]].to_string(index=False))

    closest = family_df.iloc[(family_df["savings"] - TARGET_SAVINGS).abs().argsort().iloc[0]]
    print(f"\nClosest to {TARGET_SAVINGS*100:.0f}% savings: {closest['config_name']} "
          f"{closest['channels']} -> {closest['savings']*100:.2f}%")

    grid_rows = []
    for filter_name, channels in FILTER_CHANNELS.items():
        m = measure(channels)
        macs_ratio = m["macs"] / original["macs"]
        mem_ratio = m["memory_bits"] / original["memory_bits"]
        savings = 1 - (ALPHA * macs_ratio + BETA * mem_ratio)
        grid_rows.append({"filter": filter_name, "total_channels": m["total_channels"], "savings": savings})
    grid_df = pd.DataFrame(grid_rows)

    plot(family_df, grid_df, closest)


def plot(family_df: pd.DataFrame, grid_df: pd.DataFrame, closest: pd.Series) -> None:
    import matplotlib.pyplot as plt

    ink, secondary_ink, muted, grid_color, baseline, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    series_blue, series_orange, series_aqua = "#2a78d6", "#eb6834", "#1baf7a"

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(True, color=grid_color, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(baseline)
    ax.tick_params(colors=muted, labelsize=9)

    ax.plot(family_df["total_channels"], family_df["savings"] * 100, color=series_blue,
            linewidth=2, marker="o", markersize=5, zorder=3, label="Original-structure-preserving reduction (f_i fixed)")
    label_every = max(1, len(family_df) // 10)
    for i, row in family_df.iterrows():
        if i % label_every == 0 or i == len(family_df) - 1:
            label = row["config_name"].replace("Original_fixfi_", "")
            ax.annotate(label, (row["total_channels"], row["savings"] * 100),
                        fontsize=7, color=secondary_ink, textcoords="offset points", xytext=(4, 4))

    ax.scatter(grid_df["total_channels"], grid_df["savings"] * 100, color=series_orange, s=60,
               zorder=4, marker="s", label="Section 2a grid search (Original/O2/O4/O8/O16/OF, bnnative)")
    for _, row in grid_df.iterrows():
        ax.annotate(row["filter"], (row["total_channels"], row["savings"] * 100),
                    fontsize=8, color=secondary_ink, textcoords="offset points", xytext=(5, -11))

    ax.axhline(50, color=series_aqua, linewidth=1.5, linestyle="--", zorder=2, label="50% target")
    ax.scatter([closest["total_channels"]], [closest["savings"] * 100], s=220, facecolors="none",
               edgecolors=series_aqua, linewidths=2, zorder=5,
               label=f"closest to 50%: {closest['config_name']} {closest['channels']} ({closest['savings']*100:.1f}%)")

    ax.set_xlabel("total channels (sum of all 5 stage widths)", color=secondary_ink, fontsize=10)
    ax.set_ylabel("hardware savings vs. Original (%)", color=secondary_ink, fontsize=10)
    ax.set_title("Hardware savings: Original-structure-preserving reduction (f_i fixed) vs. section 2a grid search", color=ink, fontsize=12)
    ax.legend(frameon=False, fontsize=8, labelcolor=secondary_ink, loc="lower right")

    fig.tight_layout()
    out_path = OUT_DIR / "symmetric_reduction_family.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=surface)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
