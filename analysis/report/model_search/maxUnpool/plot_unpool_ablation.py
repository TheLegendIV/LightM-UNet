"""Plots for unpool_ablation.py (reads dice_by_width.csv + stage_stats.csv from
this folder, writes exp_*.png next to them). Run in the dev container:
  docker exec lightmunet_dev python3 /workspace/LightM-UNet/analysis/report/model_search/maxUnpool/plot_unpool_ablation.py

x axis everywhere = channel divisor (1 = full-width ENet, 16 = U16). Colour = experiment,
fixed slots from the dataviz reference palette (first three slots validate
all-pairs; aqua is <3:1 on white, so every line also carries a marker shape,
a direct end label, and the underlying numbers live in the CSVs).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLIT = "val"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# exp name -> (label, colour slot, marker); control=blue, random=orange, v/4=aqua.
EXPS = {
    "control": ("control (real indices)", "#2a78d6", "o"),
    "random_shared": ("random indices (fixed, shared by all channels)", "#eb6834", "s"),
    "random_per_channel": ("random indices (fixed, per channel)", "#eb6834", "s"),
    "nearest_quarter": ("v/4 in all 4 sub-pixels", "#1baf7a", "^"),
}
STAGE_TITLE = {"up4": "decoder stage 4 (1st unpool)", "up5": "decoder stage 5 (2nd unpool)"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def aggregate(df: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Mean over seeds per (keys); lo/hi = min/max over seeds (band only shows for >1 seed)."""
    g = df.groupby(keys)[value]
    return pd.DataFrame({"mean": g.mean(), "lo": g.min(), "hi": g.max()}).reset_index()


def draw_lines(ax, agg: pd.DataFrame, divisors: list[int], theory: pd.Series | None = None) -> None:
    pos = {d: i for i, d in enumerate(divisors)}
    ends = []
    for exp, (label, colour, marker) in EXPS.items():
        sub = agg[agg["exp"] == exp].sort_values("divisor")
        if sub.empty:
            continue
        x = [pos[d] for d in sub["divisor"]]
        if (sub["hi"] - sub["lo"]).max() > 0:
            ax.fill_between(x, sub["lo"], sub["hi"], color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, sub["mean"], color=colour, linewidth=2, marker=marker, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ends.append((float(sub["mean"].iloc[-1]), colour, label.split(" (")[0]))
    if theory is not None:
        ax.plot([pos[d] for d in theory.index], theory.values, color=MUTED, linewidth=1.5, linestyle=(0, (2, 3)),
                label="independent-slot theory 1-(3/4)^C", zorder=2)
    ax.set_xticks(range(len(divisors)))
    ax.set_xticklabels([str(d) for d in divisors])
    ax.set_xlim(-0.4, len(divisors) - 0.4)
    style(ax)
    return ends


def end_labels(ax, ends, gap_px: float = 14.0) -> None:
    """Direct labels at the right end of each line, nudged apart so they never overlap."""
    fig = ax.figure
    fig.canvas.draw()
    items = sorted(((ax.transData.transform((0, y))[1], c, t) for y, c, t in ends))
    placed = []
    for ypx, colour, text in items:
        if placed and ypx - placed[-1] < gap_px:
            ypx = placed[-1] + gap_px
        placed.append(ypx)
        y_data = ax.transData.inverted().transform((0, ypx))[1]
        ax.annotate(text, xy=(ax.get_xlim()[1] - 0.38, y_data), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK2, annotation_clip=False)


def new_fig(ncols: int, w: float, h: float, nrows: int = 1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    return fig, axes


def legend_below(fig, ax, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))


def finish(fig, path: Path, title: str, subtitle: str, legend_ax) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    # Wrap the subtitle to the figure width (~11 chars per inch at 9.5pt) so it never runs off the edge.
    wrapped = textwrap.fill(subtitle, width=int(fig.get_figwidth() * 11))
    n_lines = wrapped.count("\n") + 1
    fig.text(0.01, 0.955, wrapped, ha="left", va="top", fontsize=9.5, color=INK2)
    # A narrow figure stacks the legend one entry per row instead of overflowing sideways.
    ncol = 3 if fig.get_figwidth() >= 10 else 1
    n_entries = len(legend_ax.get_legend_handles_labels()[1])
    legend_rows = -(-n_entries // ncol)
    legend_below(fig, legend_ax, ncol)
    bottom = 0.03 + 0.035 * legend_rows
    fig.tight_layout(rect=(0, bottom, 0.97, 0.9 - 0.035 * (n_lines - 1)))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path.name)


def main() -> None:
    dice = pd.read_csv(HERE / "dice_by_width.csv")
    stats = pd.read_csv(HERE / "stage_stats.csv")
    dice, stats = dice[dice["split"] == SPLIT], stats[stats["split"] == SPLIT]
    divisors = sorted(dice["divisor"].unique())
    n_cases = int(dice["n_cases"].max())
    seeds = sorted(dice[dice["exp"].str.startswith("random")]["seed"].unique())
    seed_note = f"{len(seeds)} random-index draw{'s' if len(seeds) != 1 else ''}"
    suffix = f"{n_cases} validation cases (nnU-Net fold 0), trained nets, no retraining; {seed_note}"

    # -- Dice ---------------------------------------------------------------
    fig, axes = new_fig(1, 7.2, 4.6)
    ax = axes[0][0]
    ends = draw_lines(ax, aggregate(dice, "dice", ["exp", "divisor"]), divisors)
    ax.set_xlabel("channel divisor (1 = full-width ENet, 16 = U16)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Dice, mean over 4 classes", color=INK2, fontsize=9.5)
    end_labels(ax, ends)
    finish(fig, HERE / "exp_dice.png", "Validation Dice vs. channel width, by skip ablation", suffix, ax)

    # -- per-stage panels -----------------------------------------------------
    def stage_fig(value: str, ylabel: str, fname: str, title: str, subtitle: str, theory: bool = False) -> None:
        fig, axes = new_fig(2, 11.0, 4.6)
        for ax, stage in zip(axes[0], ("up4", "up5")):
            sub = stats[stats["stage"] == stage]
            channels = sub.groupby("divisor")["skip_channels"].first()
            th = None
            if theory:
                th = sub.groupby("divisor")["pixel_fill_theory_indep"].first()
            ends = draw_lines(ax, aggregate(sub, value, ["exp", "divisor"]), divisors, th)
            ax.set_xticklabels([f"{d}\nC={int(channels[d])}" for d in divisors])
            ax.set_title(STAGE_TITLE[stage], loc="left", fontsize=10, color=INK2)
            ax.set_xlabel("channel divisor / skip channels C", color=INK2, fontsize=9.5)
            ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
            end_labels(ax, ends)
        finish(fig, HERE / fname, title, subtitle, axes[0][0])

    stage_fig("log10_main_over_out_mean", "log10( ||main|| / ||out|| )", "exp_log_main_over_out.png",
              "Skip vs. residual magnitude at the decoder add",
              f"{suffix}; 0 = equal norms, -1 = skip 10x smaller than the residual branch")
    stage_fig("pixel_fill_mean", "fraction of pixels with >= 1 non-zero skip channel", "exp_pixel_fill.png",
              "Pixel fill of the skip after unpool", suffix, theory=True)

    # -- activation stats (post out_act, per upsample block) -----------------
    cols = [("act_mean", "mean"), ("act_std", "std"), ("act_min", "min"), ("act_max", "max")]
    if stats["act_min"].abs().max() < 1e-9:  # ReLU decoder: min is 0 everywhere, a flat line carries nothing
        cols = [c for c in cols if c[0] != "act_min"]
        min_note = "min is 0 in every run (ReLU decoder) and is omitted; see stage_stats.csv"
    else:
        min_note = ""
    fig, axes = new_fig(len(cols), 4.0 * len(cols), 7.2, nrows=2)
    for r, stage in enumerate(("up4", "up5")):
        sub = stats[stats["stage"] == stage]
        for c, (value, name) in enumerate(cols):
            ax = axes[r][c]
            draw_lines(ax, aggregate(sub, value, ["exp", "divisor"]), divisors)
            ax.set_title(f"{STAGE_TITLE[stage].split(' (')[0]} - {name}", loc="left", fontsize=10, color=INK2)
            ax.set_xlabel("channel divisor", color=INK2, fontsize=9)
    finish(fig, HERE / "exp_activation_stats.png", "Activations after each upsample block (post out_act)",
           f"{suffix}. Pooled over all channels and positions. {min_note}".strip(), axes[0][0])


if __name__ == "__main__":
    main()
