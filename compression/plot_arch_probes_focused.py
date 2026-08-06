"""Focused Dice vs. params/MACs plot for just U4 (the shared reference
cell) plus every stage_4_arch_probes, stage_5_arch_probe_pairs,
stage_6_dscnoprojdense_variants, stage_7_reginterleaved_shape_variants, and
stage_8_reginterleaved_isolation config -- i.e. everything actually built
off that one reference point, with the rest of the sweep (Base/U2/U3/U6/U8/
U16/E1, stage_2_special_ops, stage_3_transfer_original) filtered out as
noise for this comparison. A horizontal dashed line at U4's own dice marks
the bar every probe is trying to clear.

Usage:
    python compression/plot_arch_probes_focused.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ABBREV_CSV = Path(__file__).resolve().parent / "config_abbreviations.csv"
U4_CONFIG_NAME = "nnUNetTrainerENet_1_naive_baseline_U4"

INK, SECONDARY_INK, MUTED, GRID, SPINE, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
STAGE_COLORS = {
    "1_naive_baseline": "#2a78d6",
    "4_arch_probes": "#eda100",
    "5_arch_probe_pairs": "#8e44ad",
    "6_dscnoprojdense_variants": "#e91e8c",
    "7_reginterleaved_shape_variants": "#00acc1",
    "8_reginterleaved_isolation": "#7cb342",
}
U4_LINE_COLOR = "#2a78d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results")
    return parser.parse_args()


def load_abbreviations() -> dict[str, str]:
    abbrev_df = pd.read_csv(ABBREV_CSV)
    return dict(zip(abbrev_df["config_name"], abbrev_df["abbrev"]))


def _plot(ax, df: pd.DataFrame, x_col: str, x_label: str, abbrev: dict[str, str], u4_dice: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.axhline(u4_dice, color=U4_LINE_COLOR, linewidth=1, linestyle="--", zorder=1,
               label=f"U4 reference (dice={u4_dice:.4f})")

    for stage, color in STAGE_COLORS.items():
        subset = df[df["stage"] == stage]
        if subset.empty:
            continue
        marker = "*" if stage == "1_naive_baseline" else "o"
        size = 220 if stage == "1_naive_baseline" else 90
        ax.scatter(subset[x_col], subset["dice"], color=color, s=size, zorder=3,
                   marker=marker, edgecolors="black", linewidths=0.6, label=stage)

    for _, row in df.iterrows():
        label = abbrev.get(row["config_name"], row["config_name"])
        ax.annotate(label, (row[x_col], row["dice"]), fontsize=8, color=SECONDARY_INK,
                   textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel(x_label, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Dice (mean of LAD/RCA/LCX/LM)", color=SECONDARY_INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="best")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1

    df = pd.read_csv(args.results_csv)
    abbrev = load_abbreviations()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    focused = df[
        (df["config_name"] == U4_CONFIG_NAME)
        | (df["stage"] == "4_arch_probes")
        | (df["stage"] == "5_arch_probe_pairs")
        | (df["stage"] == "6_dscnoprojdense_variants")
        | (df["stage"] == "7_reginterleaved_shape_variants")
        | (df["stage"] == "8_reginterleaved_isolation")
    ].dropna(subset=["params", "dice"]).copy()
    if focused.empty:
        print("No U4/stage_4/stage_5/stage_6/stage_7/stage_8 rows found in results.csv yet.")
        return 1
    u4_rows = focused[focused["config_name"] == U4_CONFIG_NAME]
    if u4_rows.empty:
        print("U4 reference row not found -- can't draw the reference line.")
        return 1
    u4_dice = float(u4_rows.iloc[0]["dice"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=SURFACE)
    _plot(ax, focused, "params", "Parameters", abbrev, u4_dice)
    ax.set_title("U4 reference + all stage_4/5/6/7 architecture probes: Dice vs. Parameters",
                 color=INK, fontsize=12)
    fig.tight_layout()
    out_path = args.out_dir / "arch_probes_focused_dice_vs_params.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")

    focused["macs"] = focused["flops"] / 2
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=SURFACE)
    _plot(ax, focused, "macs", "MACs", abbrev, u4_dice)
    ax.set_title("U4 reference + all stage_4/5/6/7 architecture probes: Dice vs. MACs",
                 color=INK, fontsize=12)
    fig.tight_layout()
    out_path = args.out_dir / "arch_probes_focused_dice_vs_macs.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")

    cols = ["config_name", "stage", "params", "flops", "dice", "converged_flag"]
    printable = focused[cols].copy()
    printable.insert(0, "abbrev", printable["config_name"].map(lambda c: abbrev.get(c, c)))
    printable["dice_vs_u4"] = printable["dice"] - u4_dice
    printable = printable.sort_values("dice", ascending=False)
    pd.set_option("display.width", 200)
    print()
    print(printable.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
