"""Stage 1 investigation: ENet official (authors' Torch code) vs this repo's
PReLU + max-unpool ENet, naive channel-width sweep.

Same plot style as the Stage 1 figures in nas_search_narrative.ipynb (styling
helpers come from nas_stages.py). Data: official_vs_maxunpool_results.csv, a
snapshot of compression/results.csv at commit a9c81dd1c6 -- the
`1_naive_baseline_official_*` rows are no longer in the live results.csv.

Run: python plot_official_vs_maxunpool.py   (needs pandas + matplotlib)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # analysis/report/model_search
import nas_stages as ns  # noqa: E402

df = pd.read_csv(HERE / "official_vs_maxunpool_results.csv")
SERIES = [
    ("maxunpool", "PReLU, Max-Unpool Upsampling (repo variant)", ns.SERIES_COLORS[2], "s"),
    ("official", "PReLU, Official ENet (3x3 transposed conv)", ns.SERIES_COLORS[1], "o"),
]

for metric, xlabel in [("params", "Parameters"), ("macs", "MACs"), ("mem_elements", "Activation Buffer Elements")]:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=ns.SURFACE)
    ns.style_axes(ax)
    ax.set_xscale("log")
    for variant, label, color, marker in SERIES:
        s = df[df["variant"] == variant].sort_values(metric)
        ax.plot(s[metric], s["dice"], color=color, linewidth=2, marker=marker,
                markersize=6, markeredgecolor=ns.SURFACE, markeredgewidth=1, zorder=3,
                label=label)

    ns.style_title_and_labels(
        ax,
        f"Dice vs. {xlabel} for Two ENet variations at Channel Widths x1/16, x1/8, x1/4, x1/2, x1.",
        xlabel,
    )
    fig.canvas.draw()
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_color(ns.INK)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_color(ns.INK)
    ax.yaxis.label.set_fontsize(11)
    ns.style_legend(ax)
    fig.tight_layout()
    out = HERE / f"dice_vs_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
    print(f"Wrote {out}")
    plt.close(fig)
