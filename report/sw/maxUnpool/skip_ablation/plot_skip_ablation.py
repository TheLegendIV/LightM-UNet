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
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
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
def finish(ax, title: str, ylabel: str, legend_loc: str, ncol: int = 1) -> None:
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
    handles, labels = ax.get_legend_handles_labels()
    band = getattr(ax, "random_band", None)
    if band is not None:  # one legend entry: gray min-max band under the mean line
        handles = [(band, h) if lab == RANDOM_LABEL else h for h, lab in zip(handles, labels)]
    ax.legend(handles, labels, loc=legend_loc, ncol=ncol, frameon=True, facecolor=ns.SURFACE, edgecolor="#c3c2b7",
              fontsize=6.5, labelcolor=ns.SECONDARY_INK, handlelength=3.6, borderpad=0.4, labelspacing=0.3)
    for t in ax.get_legend().get_texts():
        t.set_fontweight("bold")
    ax.figure.tight_layout()


RANDOM_LABEL = "Random Indices"
mu =wide[wide["curve"] == "maxunpool"].sort_values("divisor").reset_index(drop=True)
rand_cols = sorted(c for c in mu.columns if c.startswith("random_pc_s") and mu[c].notna().all())
mu["random"] = mu[rand_cols].mean(axis=1)
mu["random_min"], mu["random_max"] = mu[rand_cols].min(axis=1), mu[rand_cols].max(axis=1)
for col in ("noskip", "random", "random_min", "random_max", "nearest_quarter", "nearest_half"):
    mu[f"d_{col}"] = (mu[col] - mu["control"]) / mu["control"] * 100  # % change vs control
    mu[f"dpts_{col}"] = (mu[col] - mu["control"]) * 100  # Dice points
bl = wide[wide["curve"] == "bilinear"].sort_values("divisor")
assert list(bl["divisor"]) == list(mu["divisor"]), "bilinear and max-unpool widths differ"
mu["bilinear"], mu["bilinear_noskip"] = bl["control"].values, bl["noskip"].values
for col in ("bilinear", "bilinear_noskip"):  # change vs the MaxUnpool control, same baseline as the arms
    mu[f"d_{col}"] = (mu[col] - mu["control"]) / mu["control"] * 100
    mu[f"dpts_{col}"] = (mu[col] - mu["control"]) * 100
keep = ["divisor", "control", "noskip", "random", "random_min", "random_max", "nearest_quarter", "nearest_half",
        "bilinear", "bilinear_noskip"]
mu[keep + [c for c in mu.columns if c.startswith(("d_", "dpts_"))]].to_csv(HERE / "maxunpool_arms.csv", index=False)
print(f"random arm = mean of {rand_cols}")

# (column, label, colour, linestyle, linewidth, marker, markersize); control is drawn as a
# wider symbol-less underlay so arms that coincide with it stay visible.
ARMS = [
    ("noskip", "MaxUnpool No Skip", ns.INK, (0, (3, 2)), 1.0, "+", 6),
    ("random", RANDOM_LABEL, ns.SERIES_COLORS[1], "-", 1.0, "x", 5),
    ("nearest_quarter", "v/4", ns.SERIES_COLORS[4], "-", 1.0, (6, 2, 0), 6),  # asterisk
    ("nearest_half", "v/2", ns.SERIES_COLORS[5], "-", 1.0, "o", 4),
]
CONTROL = ("control", "MaxUnpool", ns.SERIES_COLORS[2], "-", 2.4)


def draw(ax, col, label, color, ls, lw, marker, msize, delta: bool) -> None:
    y = mu[f"d_{col}"] if delta else mu[col]
    kw = dict(color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=msize, markeredgecolor=color,
              markerfacecolor=color, markeredgewidth=1.2, label=label)
    ax.plot(mu["divisor"], y, zorder=3, **kw)
    if col == "random":  # gray band spanning min-max over the random draws
        lo = mu["d_random_min"] if delta else mu["random_min"]
        hi = mu["d_random_max"] if delta else mu["random_max"]
        ax.random_band = ax.fill_between(mu["divisor"], lo, hi, color="#898781", alpha=0.35, linewidth=0,
                                         zorder=1)


fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
ax.plot(mu["divisor"], mu["control"], color=CONTROL[2], linewidth=CONTROL[4], zorder=2, label=CONTROL[1])
draw(ax, *ARMS[0], delta=False)  # MaxUnpool No Skip
# Bilinear nets: only control and zeroed skip apply (no indices to randomise, no nearest swap).
BLUE = ns.SERIES_COLORS[0]
ax.plot(bl["divisor"], bl["control"], color=BLUE, linewidth=CONTROL[4], zorder=2, label="Bilinear")
ax.plot(bl["divisor"], bl["noskip"], color=BLUE, linestyle=ARMS[0][3], linewidth=1.0, marker="+", markersize=6,
        markeredgecolor=BLUE, markeredgewidth=1.2, zorder=3, label="Bilinear No Skip")
for arm in ARMS[1:]:
    draw(ax, *arm, delta=False)
ax.set_ylim(0.2, 1.08)  # headroom for the two-column legend
finish(ax, "Ablation of MaxUnpool at Various Channel Widths", "Dice", "upper left", ncol=2)
out = HERE / "dice_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
plt.close(fig)

class SideBySide(tuple):
    """Legend handle drawn as its parts next to each other (a plain tuple overlays them)."""


def add_legend(ax, loc: str, ncol: int = 1, extra=None, anchor=None, handlelength: float = 3.6,
               columnspacing: float = 2.0) -> None:
    handles, labels = ax.get_legend_handles_labels()
    band = getattr(ax, "random_band", None)
    if band is not None:  # one legend entry: gray min-max band under the mean line
        handles = [(band, h) if lab == RANDOM_LABEL else h for h, lab in zip(handles, labels)]
    if extra:  # bilinear's line and the out-of-range triangle share one legend entry, side by side
        handles = [SideBySide((h, extra[0])) if lab == "Bilinear" else h for h, lab in zip(handles, labels)]
    ax.legend(handles, labels, loc=loc, ncol=ncol, frameon=True, facecolor=ns.SURFACE, edgecolor="#c3c2b7",
              handler_map={SideBySide: HandlerTuple(ndivide=None, pad=0.2)},
              fontsize=6.5, labelcolor=ns.SECONDARY_INK, handlelength=handlelength, columnspacing=columnspacing,
              borderpad=0.4, labelspacing=0.3, **({"bbox_to_anchor": anchor} if anchor else {}))
    for t in ax.get_legend().get_texts():
        t.set_fontweight("bold")


LIM = 12  # y limit (%) of the delta panels; v/4 reaches -9.0% at 1/8, so +-12 keeps every MaxUnpool arm in frame
TRI = 7  # out-of-range triangle size (pt)
OOR_HANDLE = Line2D([], [], marker="^", linestyle="none", color=BLUE, markersize=TRI)  # out-of-range triangle


def off_scale(ax, x, y, color) -> None:
    """Points beyond +-LIM: a small triangle at the frame edge pointing the way the value went, with the
    percentage next to it (below it at the top edge, above it at the bottom edge)."""
    lo, hi = ax.get_ylim()
    ms = TRI  # marker size (pt); a triangle marker's tip is ms/2 from its centre, so shift the centre by that
    xaxis = ax.get_xaxis_transform()  # x in data, y as a fraction of the axes height
    for xi, yi in zip(x, y):
        if yi > hi:  # tip touches the top frame line
            ax.plot(xi, 1, marker="^", markersize=ms, markeredgewidth=0, color=color, linestyle="none",
                    transform=offset_copy(xaxis, fig=ax.figure, y=-ms / 2, units="points"), clip_on=False, zorder=4)
            ax.annotate(f"+{yi:.0f}%", xy=(xi, 1), xycoords=xaxis, xytext=(0, -(ms + 2)), textcoords="offset points",
                        color=color, fontsize=6.5, fontweight="bold", ha="center", va="top", zorder=4)
        elif yi < lo:  # tip touches the bottom frame line
            ax.plot(xi, 0, marker="v", markersize=ms, markeredgewidth=0, color=color, linestyle="none",
                    transform=offset_copy(xaxis, fig=ax.figure, y=ms / 2, units="points"), clip_on=False, zorder=4)
            ax.annotate(f"−{abs(yi):.0f}%", xy=(xi, 0), xycoords=xaxis, xytext=(0, ms + 2),
                        textcoords="offset points", color=color, fontsize=6.5, fontweight="bold",
                        ha="center", va="bottom", zorder=4)


def delta_panel(ax, ylim=(-LIM, LIM)) -> None:
    """% change vs the MaxUnpool control on a plain linear axis; values outside `ylim` are drawn as
    labelled triangles at the frame edge (only the bilinear control leaves the frame; the bilinear
    no-skip run is left out of this panel, see dice_vs_width.png)."""
    ax.axhline(0, color=CONTROL[2], linewidth=CONTROL[4], zorder=2, label=CONTROL[1])
    draw(ax, *ARMS[0], delta=True)  # MaxUnpool No Skip
    ax.plot(mu["divisor"], mu["d_bilinear"], color=BLUE, linewidth=CONTROL[4], zorder=2, label="Bilinear")
    for arm in ARMS[1:]:
        draw(ax, *arm, delta=True)
    ax.set_ylim(*ylim)
    off_scale(ax, mu["divisor"], mu["d_bilinear"], BLUE)
    for arm in ARMS:
        off_scale(ax, mu["divisor"], mu[f"d_{arm[0]}"], arm[2])


fig, ax = plt.subplots(figsize=(5.0, 3.8), facecolor=ns.SURFACE)
ns.style_axes(ax)
delta_panel(ax, ylim=(-18, LIM))  # extra room below the lines for the legend
finish(ax, "Ablation of MaxUnpool at Various Channel Widths", "Dice Change (%)", "lower right")
add_legend(ax, "lower right", ncol=2, extra=[OOR_HANDLE], handlelength=3.2, columnspacing=1.0)
out = HERE / "dice_delta_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
print(mu[keep].round(4).to_string(index=False))
print(mu[["divisor"] + [c for c in mu.columns if c.startswith(("d_", "dpts_"))]].round(2).to_string(index=False))


# --- Combined figure: absolute Dice (top) over % change (bottom), shared x axis --
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.0, 6.6), facecolor=ns.SURFACE)
ns.style_axes(ax1)
ns.style_axes(ax2)

# top: same content as dice_vs_width.png
ax1.plot(mu["divisor"], mu["control"], color=CONTROL[2], linewidth=CONTROL[4], zorder=2, label=CONTROL[1])
draw(ax1, *ARMS[0], delta=False)
ax1.plot(bl["divisor"], bl["control"], color=BLUE, linewidth=CONTROL[4], zorder=2, label="Bilinear")
ax1.plot(bl["divisor"], bl["noskip"], color=BLUE, linestyle=ARMS[0][3], linewidth=1.0, marker="+", markersize=6,
         markeredgecolor=BLUE, markeredgewidth=1.2, zorder=3, label="Bilinear No Skip")
for arm in ARMS[1:]:
    draw(ax1, *arm, delta=False)
ax1.set_ylim(0.2, 1.12)  # headroom for the two-column legend
ax1.set_ylabel("Dice")
add_legend(ax1, "upper left", ncol=2, extra=[OOR_HANDLE])

# bottom: same content as dice_delta_vs_width.png
delta_panel(ax2)
ax2.set_ylabel("Dice Change (%)")  # no legend here: the top panel's legend covers every line
ax2.set_xlabel("Channel Width Multiplier")

# shared x axis: set once, on the bottom panel (applies to both)
ax2.set_xscale("log", base=2)
ax2.set_xticks(divs)
ax2.set_xticklabels([f"1/{int(d)}" if d > 1 else "1" for d in divs])
ax2.minorticks_off()
ax2.invert_xaxis()  # narrowest width (1/16) on the left
fig.suptitle("Skip Branch Ablation Across Channel Widths", color=ns.INK, fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.canvas.draw()
for a in (ax1, ax2):
    for t in a.get_xticklabels() + a.get_yticklabels():
        t.set_fontweight("bold")
    for lab in (a.xaxis.label, a.yaxis.label):
        lab.set_fontweight("bold"); lab.set_color(ns.INK); lab.set_fontsize(11)
out = HERE / "dice_ablation_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=ns.SURFACE)
print(f"Wrote {out}")
plt.close(fig)
