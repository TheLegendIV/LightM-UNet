"""Plots for the Stage 1 inference-time skip ablation (skip_ablation.py). Reads
skip_ablation_results.csv (test split). Needs pandas + matplotlib. Writes:
  dice_drop_vs_width.png/.csv  % Dice decrease from zeroing the skip, max-unpool vs bilinear
  dice_vs_width.png            absolute Dice of the five skip-branch arms on the max-unpool nets
  dice_delta_vs_width.png      the same arms as Dice change vs control (points)
  maxunpool_arms.csv           the table behind the last two
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



# --- Five skip-branch arms on the max-unpool nets -----------------------------
def finish(ax, title: str, ylabel: str, legend_loc: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(divs)
    ax.set_xticklabels([f"1/{int(d)}" if d > 1 else "1" for d in divs])
    ax.minorticks_off()
    ax.invert_xaxis()  # narrowest width (1/16) on the left
    ns.style_title_and_labels(ax, title, "Channel Width Multiplier", ylabel)
    ax.figure.canvas.draw()
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_fontweight("bold"); lab.set_color(ns.INK); lab.set_fontsize(11)
    ax.legend(loc=legend_loc, frameon=True, facecolor=ns.SURFACE, edgecolor="#c3c2b7", fontsize=6.5,
              labelcolor=ns.SECONDARY_INK, handlelength=3.6, borderpad=0.4, labelspacing=0.3)
    for t in ax.get_legend().get_texts():
        t.set_fontweight("bold")
    ax.figure.tight_layout()


mu = wide[wide["curve"] == "maxunpool"].sort_values("divisor").reset_index(drop=True)
rand_cols = sorted(c for c in mu.columns if c.startswith("random_pc_s") and mu[c].notna().all())
mu["random"] = mu[rand_cols].mean(axis=1)
mu["random_min"], mu["random_max"] = mu[rand_cols].min(axis=1), mu[rand_cols].max(axis=1)
for col in ("noskip", "random", "random_min", "random_max", "nearest_quarter", "nearest_half"):
    mu[f"d_{col}"] = (mu[col] - mu["control"]) * 100  # points
keep = ["divisor", "control", "noskip", "random", "random_min", "random_max", "nearest_quarter", "nearest_half"]
mu[keep + [c for c in mu.columns if c.startswith("d_")]].to_csv(HERE / "maxunpool_arms.csv", index=False)
print(f"random arm = mean of {rand_cols}")

# (column, label, colour, linestyle, linewidth); control is drawn as a wider underlay so
# arms that coincide with it stay visible.
ARMS = [
    ("noskip", "No Skip (0s)", ns.INK, (0, (3, 2)), 1.0),
    ("random", "Random Indices", ns.SERIES_COLORS[1], "-", 1.0),
    ("nearest_quarter", "v/4 (Matched Sum)", ns.SERIES_COLORS[4], "-", 1.0),
    ("nearest_half", "v/2 (Matched L2 Norm)", ns.SERIES_COLORS[0], "-", 1.0),
]
CONTROL = ("control", "MaxUnpool", ns.SERIES_COLORS[2], "-", 2.4)


def draw(ax, col, label, color, ls, lw, delta: bool) -> None:
    y = mu[f"d_{col}"] if delta else mu[col]
    kw = dict(color=color, linestyle=ls, linewidth=lw, marker="+", markersize=5, markeredgecolor=color,
              markeredgewidth=1.2, label=label)
    if col == "random":  # min-max over the random draws
        lo = mu["d_random_min"] if delta else mu["random_min"]
        hi = mu["d_random_max"] if delta else mu["random_max"]
        ax.errorbar(mu["divisor"], y, yerr=[y - lo, hi - y], capsize=2, elinewidth=0.8, zorder=3, **kw)
    else:
        ax.plot(mu["divisor"], y, zorder=3, **kw)


fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
ax.plot(mu["divisor"], mu["control"], color=CONTROL[2], linewidth=CONTROL[4], marker="+", markersize=6,
        markeredgecolor=CONTROL[2], markeredgewidth=1.2, zorder=2, label=CONTROL[1])
for arm in ARMS:
    draw(ax, *arm, delta=False)
ax.set_ylim(0.2, 0.88)
finish(ax, "Skip Branch Ablation of MaxUnpool Nets", "Dice", "lower right")
out = HERE / "dice_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
ax.axhline(0, color=CONTROL[2], linewidth=CONTROL[4], zorder=2, label=CONTROL[1])
for arm in ARMS:
    draw(ax, *arm, delta=True)
finish(ax, "Dice Change vs MaxUnpool", "Dice Change (pts)", "lower right")
out = HERE / "dice_delta_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
print(mu[keep].round(4).to_string(index=False))
print(mu[["divisor"] + [c for c in mu.columns if c.startswith("d_")]].round(2).to_string(index=False))
