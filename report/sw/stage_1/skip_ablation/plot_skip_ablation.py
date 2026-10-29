"""Plot the Dice drop from removing the decoder skip branch at inference,
vs channel width, for the max-unpool and bilinear (PReLU) Stage 1 curves.
Reads skip_ablation_results.csv (from skip_ablation.py), writes dice_drop_vs_width.png
and dice_drop_vs_width.csv. Needs pandas + matplotlib.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # report/sw
import nas_stages as ns  # noqa: E402

df = pd.read_csv(HERE / "skip_ablation_results.csv")
df = df[df["split"] == "test"]
wide = df.pivot_table(index=["curve", "divisor"], columns="exp", values="dice").reset_index()
wide["drop"] = wide["control"] - wide["noskip"]
wide.to_csv(HERE / "dice_drop_vs_width.csv", index=False)

SERIES = [
    ("maxunpool", "Max-Unpool", ns.SERIES_COLORS[2], "s"),
    ("bilinear", "Bilinear Upsample", ns.SERIES_COLORS[0], "o"),
]
fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=ns.SURFACE)
ns.style_axes(ax)
for curve, label, color, marker in SERIES:
    s = wide[wide["curve"] == curve].sort_values("divisor")
    ax.plot(s["divisor"], s["drop"], color=color, linewidth=2, marker=marker, markersize=7,
            markeredgecolor=ns.SURFACE, markeredgewidth=1, zorder=3, label=label)
ax.set_xscale("log", base=2)
divs = sorted(wide["divisor"].unique())
ax.set_xticks(divs)
ax.set_xticklabels([f"1/{int(d)}" if d > 1 else "1" for d in divs])
ax.minorticks_off()
ax.axhline(0, color=ns.MUTED, linewidth=0.8, zorder=1)
ns.style_title_and_labels(ax, "Dice Drop When Removing the Upsample Skip Branch",
                          "Channel Width", "Dice (Control) - Dice (No Skip)")
fig.canvas.draw()
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold")
for lab in (ax.xaxis.label, ax.yaxis.label):
    lab.set_fontweight("bold"); lab.set_color(ns.INK); lab.set_fontsize(11)
ns.style_legend(ax, loc="best")
for t in ax.get_legend().get_texts():
    t.set_fontweight("bold")
fig.tight_layout()
out = HERE / "dice_drop_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
print(wide.round(4).to_string(index=False))
