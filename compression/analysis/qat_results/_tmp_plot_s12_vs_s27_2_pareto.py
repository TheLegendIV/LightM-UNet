"""TEMPORARY quick-look script -- overlays S12 and S27_2's own forced-DSP
70%-LUT alpha-sweep dice-vs-latency Pareto points (both already computed by
plot_forcedsp_lut70_alpha_sweep.py) on ONE figure for a fast side-by-side
comparison. Not part of the established analysis pipeline naming convention --
delete once no longer needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_forcedsp_lut70_alpha_sweep import (  # noqa: E402
    INK, SECONDARY_INK, MUTED, SURFACE, REPO_ROOT,
    _style_axes, load_final_dice, load_ilp_latency,
)

DEFAULT_CSV = REPO_ROOT / "compression" / "results.csv"
OUT_PATH = Path(__file__).resolve().parent / "out" / "_tmp_s12_vs_s27_2_pareto.png"

ARCHS = {
    "S12": {"config_prefix": "12_separable_dense_relu", "ilp_dir_prefix": "S12", "marker": "o"},
    "S27_2": {"config_prefix": "27_2_reg_trailing", "ilp_dir_prefix": "S27_2", "marker": "^"},
}
ARCH_COLOR = {"S12": "#2a78d6", "S27_2": "#c44536"}


def main() -> int:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    _style_axes(ax)

    for arch_name, spec in ARCHS.items():
        dice = load_final_dice(DEFAULT_CSV, spec["config_prefix"], "dice")
        ilp_summary = REPO_ROOT / "compression" / "hawq" / "artifacts" / f"{spec['ilp_dir_prefix']}_ILP_outputs_perlayer_forcedsp_lut70" / "summary.csv"
        latency = load_ilp_latency(ilp_summary)
        alphas = sorted(set(dice) & set(latency))
        if not alphas:
            print(f"Note: no data for {arch_name} -- skipping.")
            continue
        xs = [latency[a] for a in alphas]
        ys = [dice[a] for a in alphas]
        color = ARCH_COLOR[arch_name]
        ax.plot(xs, ys, color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=2)
        ax.scatter(xs, ys, color=color, marker=spec["marker"], s=110, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2, label=arch_name)
        for a, x, y in zip(alphas, xs, ys):
            ax.annotate(f"{a}", (x, y), xytext=(7, 5), textcoords="offset points",
                        color=color, fontsize=8, fontweight="bold")

    ax.set_xlabel("ILP-predicted latency (ms @ 100MHz)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("best-checkpoint dice (checkpoint_best.pth)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        "S12 vs. S27_2: accuracy/latency Pareto at a fixed 70% LUT budget (forced-DSP, bits {4,6,8})\n"
        "labels are alpha; TEMPORARY quick-look plot",
        color=INK, fontsize=11,
    )
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7",
              fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
