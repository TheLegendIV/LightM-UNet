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
wide["drop"] = (wide["control"] - wide["noskip"]) / wide["control"] * 100  # % decrease
wide.to_csv(HERE / "dice_drop_vs_width.csv", index=False)

SERIES = [
    ("maxunpool", "Max-Unpool", ns.SERIES_COLORS[2], "+"),
    ("bilinear", "Bilinear", ns.SERIES_COLORS[0], "x"),
]
fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
for curve, label, color, marker in SERIES:
    s = wide[wide["curve"] == curve].sort_values("divisor")
    ax.plot(s["divisor"], s["drop"], color=color, linewidth=1.2, marker=marker, markersize=5,
            markeredgecolor=color, markeredgewidth=1.2, zorder=3, label=label)
ax.set_xscale("log", base=2)
divs = sorted(wide["divisor"].unique())
ax.set_xticks(divs)
ax.set_xticklabels([f"1/{int(d)}" if d > 1 else "1" for d in divs])
ax.minorticks_off()
ax.invert_xaxis()  # narrowest width (1/16) on the left
ax.axhline(0, color=ns.MUTED, linewidth=0.8, zorder=1)
ns.style_title_and_labels(
    ax, "Ablation of Skip Branch in Upsample Bottlenecks",
    "Channel Width Multiplier", "Dice Decrease (%)")
fig.canvas.draw()
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold")
for lab in (ax.xaxis.label, ax.yaxis.label):
    lab.set_fontweight("bold"); lab.set_color(ns.INK); lab.set_fontsize(11)
ax.legend(loc="best", frameon=True, facecolor=ns.SURFACE, edgecolor="#c3c2b7", fontsize=6.5,
          labelcolor=ns.SECONDARY_INK, handlelength=3.6, borderpad=0.4, labelspacing=0.3)
for t in ax.get_legend().get_texts():
    t.set_fontweight("bold")
fig.tight_layout()
out = HERE / "dice_drop_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
print(wide.round(4).to_string(index=False))

# --- Second figure: absolute Dice, with and without the skip branch ----------
fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
for curve, label, color, marker in SERIES:
    s = wide[wide["curve"] == curve].sort_values("divisor")
    ax.plot(s["divisor"], s["control"], color=color, linewidth=1.2, marker=marker, markersize=5,
            markeredgecolor=color, markeredgewidth=1.2, zorder=3, label=f"{label}, Skip")
    ax.plot(s["divisor"], s["noskip"], color=color, linewidth=1.2, linestyle=(0, (3, 2)), marker=marker,
            markersize=5, markeredgecolor=color, markeredgewidth=1.2,
            zorder=3, label=f"{label}, No Skip")
ax.set_xscale("log", base=2)
ax.set_xticks(divs)
ax.set_xticklabels([f"1/{int(d)}" if d > 1 else "1" for d in divs])
ax.minorticks_off()
ax.invert_xaxis()  # narrowest width (1/16) on the left
ax.set_ylim(0.2, 0.88)
ns.style_title_and_labels(ax, "Dice With and Without Skip Branch", "Channel Width Multiplier", "Dice")
fig.canvas.draw()
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold")
for lab in (ax.xaxis.label, ax.yaxis.label):
    lab.set_fontweight("bold"); lab.set_color(ns.INK); lab.set_fontsize(11)
ax.legend(loc="lower right", frameon=True, facecolor=ns.SURFACE, edgecolor="#c3c2b7", fontsize=6.5,
          labelcolor=ns.SECONDARY_INK, handlelength=3.6, borderpad=0.4, labelspacing=0.3)
for t in ax.get_legend().get_texts():
    t.set_fontweight("bold")
fig.tight_layout()
out = HERE / "dice_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
