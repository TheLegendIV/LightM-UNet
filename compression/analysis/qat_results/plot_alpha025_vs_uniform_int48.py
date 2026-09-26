"""Separate, focused variant of plot_forcedsp_lut70_alpha_sweep.py's own
dice-vs-latency Pareto plot: dice vs. LUT CONSUMPTION (%) for four points --
alpha=0.25's own real per-layer ILP solve ("Mixed") and the naive
uniform-quantization baseline's INT4/INT6/INT8 points (folding pinned to
alpha=0.25's own real ILP solve, from compression/hawq/
uniform_bits_same_folding.py's run -- same source plot_uniform_vs_ilp_
overlay.py already uses). The rest of the alpha sweep is dropped entirely
(not just greyed out) -- this plot exists purely to compare Mixed vs.
INT4/INT6/INT8 at a glance.

Usage:
    python compression/analysis/qat_results/plot_alpha025_vs_uniform_int48.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_forcedsp_lut70_alpha_sweep import (  # noqa: E402
    INK, SECONDARY_INK, SURFACE, ALPHA_COLORS, REPO_ROOT,
    _style_axes, load_epoch_trend, load_ilp_lut_pct, load_fp32_baseline,
)
from plot_uniform_vs_ilp_overlay import (  # noqa: E402
    UNIFORM_LUT_PCT, UNIFORM_COLOR,
)

DEFAULT_CSV = REPO_ROOT / "compression" / "results.csv"
OUT_PATH = Path(__file__).resolve().parent / "out" / "12_dense_relu_warmstart150ep_alpha025_vs_uniform_int48.png"

CONFIG_PREFIX = "12_dense_relu_warmstart150ep"
ILP_DIR_PREFIX = "12_dense_relu_warmstart150ep"
HIGHLIGHT_ALPHA = 0.25


def load_uniform_dice_epoch15(csv_path: Path) -> dict[int, float]:
    """The FIXED epoch15 snapshot row (not checkpoint_best.pth's own
    EMA-best row) for each uniform bit-width -- avoids the EMA-best-
    plateaued-early quirk seen on INT6 (best@epoch1, dice 0.7603, vs. its
    own real epoch15 snapshot, dice 0.7805)."""
    import csv

    names = {
        bits: f"nnUNetTrainerLayerQuantENet_12_dense_relu_warmstart150ep_perlayer_"
              f"12_dense_relu_warmstart150ep_uniform_int{bits}_samefolding_alpha0.25_ft15ep_epoch15"
        for bits in (4, 6, 8)
    }
    with open(csv_path, newline="") as f:
        rows = {row["config_name"]: row for row in csv.DictReader(f)}
    return {bits: float(rows[name]["dice"]) for bits, name in names.items() if name in rows}


def main() -> int:
    import matplotlib.pyplot as plt

    ilp_summary = REPO_ROOT / "MILP" / "artifacts" / "archive" / f"{ILP_DIR_PREFIX}_ILP_outputs_perlayer_forcedsp_lut70" / "summary.csv"
    ilp_epoch_trend = load_epoch_trend(DEFAULT_CSV, CONFIG_PREFIX, "dice")
    ilp_lut_pct = load_ilp_lut_pct(ilp_summary)
    uniform_dice = load_uniform_dice_epoch15(DEFAULT_CSV)
    fp32_dice = load_fp32_baseline(DEFAULT_CSV, CONFIG_PREFIX, "dice")

    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    _style_axes(ax)

    if fp32_dice is not None:
        ax.axhline(fp32_dice, color=INK, linestyle=":", linewidth=1.2, zorder=1,
                   label=f"FP32 baseline ({fp32_dice:.4f})")

    ax.axvline(100.0, color=INK, linewidth=1.2, linestyle="-", zorder=1, alpha=0.6)

    # -- alpha=0.25's own real per-layer ILP solve ("Mixed"), epoch15 fixed snapshot. --
    mixed_x, mixed_y = ilp_lut_pct[HIGHLIGHT_ALPHA], ilp_epoch_trend[HIGHLIGHT_ALPHA][15]
    ax.scatter([mixed_x], [mixed_y], color=ALPHA_COLORS.get(HIGHLIGHT_ALPHA), s=130, zorder=5,
               marker="o", edgecolor=SURFACE, linewidth=1.4)
    ax.annotate("Mixed", (mixed_x, mixed_y), xytext=(8, 8),
                textcoords="offset points", color=ALPHA_COLORS.get(HIGHLIGHT_ALPHA), fontsize=9)

    # -- Naive uniform baseline, INT4/INT6/INT8, folding pinned to alpha=0.25.
    # All markers are solid filled circles; red fill signals a bit-width that
    # doesn't fit the chip's LUT budget (INT8 only, at this folding). --
    for bits in (4, 6, 8):
        if bits not in uniform_dice:
            continue
        dice = uniform_dice[bits]
        lut_pct = UNIFORM_LUT_PCT[bits]
        fits = lut_pct <= 100.0
        color = UNIFORM_COLOR if fits else "#c0392b"
        ax.scatter([lut_pct], [dice], color=color, s=130, zorder=5, linewidth=2,
                   marker="o", edgecolor=SURFACE)
        ax.annotate(f"INT{bits}", (lut_pct, dice), xytext=(8, 8),
                    textcoords="offset points", color=color, fontsize=9)

    ax.set_xlabel("LUT Consumption (%)", color=INK, fontsize=10, fontweight="bold")
    ax.set_ylabel("Dice", color=INK, fontsize=10, fontweight="bold")
    ax.set_title(
        f"α={HIGHLIGHT_ALPHA} ILP Allocation vs. Naive Uniform Quantization (INT4/INT6/INT8)",
        color=INK, fontsize=10.5, fontweight="bold",
    )
    ax.set_ylim(top=0.8)
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7",
              fontsize=8.5, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
