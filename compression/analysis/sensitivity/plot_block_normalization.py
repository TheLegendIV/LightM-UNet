"""Given a block_sensitivity_*.json file (sensitivity.py/block_sensitivity.py
output), plots one bit-width/metric's per-block scores four ways, all on
the same block ordering, so the effect of ilp_search.py's _normalize() --
and of log-transforming before it -- is visible directly on real data
instead of reasoned about in the abstract:

  1. raw          -- the untransformed value, log-x lollipop (spans several
                      orders of magnitude across 20-40 blocks; a linear
                      axis would flatten everything but the top block).
  2. log -> min-max -- log10(v), then plain min-max to [0,1]. Answers
                      "what if we log first?": turns MULTIPLICATIVE gaps
                      into additive ones, so a block doesn't dominate the
                      axis just for being big in absolute terms.
  3. min-max      -- ilp_search.py's _normalize(..., robust_pct=0.0), i.e.
                      its default behavior: (v-min)/(max-min).
  4. robust p-100+p -- ilp_search.py's _normalize(..., robust_pct=p):
                      anchors to the [p, 100-p] percentiles and clips
                      beyond them.

_percentile/_normalize_1d below are a deliberate copy of
compression/hawq/ilp_search.py's own _percentile/_normalize (same
algorithm, kept in sync by hand) rather than an import -- ilp_search.py
pulls in `pulp` at module scope for its solver, which this plotting-only
script has no other reason to require.

Empirically (see the printed diagnostic), log-transforming does NOT fix
outlier fragility -- it relocates it. Min-max in ANY space is anchored by
whichever two points are most extreme in THAT space: linear min-max lets
one unusually-LARGE block (a big multiplicative outlier) eat a large
fraction of the [0,1] range: bin/download the printed "wasted range"
figure for the concrete number. Log min-max instead lets one unusually-
SMALL block (a big DOWNWARD multiplicative outlier -- e.g. a near-dead
1x1 conv) eat a comparable fraction at the bottom. Neither transform
alone is robust; only percentile-clipping (or a median/MAD-based scale)
protects both tails at once -- see ilp_search.py's --robust-normalize-pct.

Usage:
    python compression/analysis/sensitivity/plot_block_normalization.py \\
        compression/hawq/artifacts/block_sensitivity_12_separable_dense_relu.json

    python compression/analysis/sensitivity/plot_block_normalization.py \\
        compression/hawq/artifacts/block_sensitivity_27_2_reg_trailing.json \\
        --metric sensitivity_a --bits 8 --robust-pct 10 --highlight down1 up5
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

# dataviz skill's validated categorical palette (fixed order, not cycled --
# see compression/plot_cost_relationships.py for the same convention).
INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
OTHER_COLOR = "#2a78d6"      # slot 1 blue -- every non-highlighted block
HIGHLIGHT_COLOR = "#eb6834"  # slot 2 orange -- every highlighted block, ONE shared color

# dataviz skill's palette note: small multiples only validate the CVD/normal-
# vision floors across the first THREE categorical slots (--pairs all), so
# --highlight is capped at 2 colors total (other + one highlight hue) no
# matter how many block NAMES are passed -- all of them share the same
# highlight color, distinguished from each other by their row label and
# their own value annotation, not by hue. Was previously one hue per name
# (up to 2); generalized to an arbitrary-length group so e.g. "these 4
# blocks are pinned to 8-bit" reads as one call-out group instead of
# silently reusing colors past the 2-name mark.


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile's default
    'linear' method). `sorted_values` must already be sorted ascending.
    pct=0 returns the min, pct=100 returns the max."""
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    lo_idx, hi_idx = math.floor(rank), math.ceil(rank)
    if lo_idx == hi_idx:
        return sorted_values[lo_idx]
    frac = rank - lo_idx
    return sorted_values[lo_idx] + frac * (sorted_values[hi_idx] - sorted_values[lo_idx])


def _normalize_1d(values: dict[str, float], robust_pct: float = 0.0) -> dict[str, float]:
    """Min-max scale a flat {name: value} dict to [0,1]. robust_pct=0.0
    (default) is plain min-max; robust_pct=p anchors to the [p, 100-p]
    percentiles and clips beyond them -- mirrors ilp_search.py's
    _normalize()."""
    vals = sorted(values.values())
    lo = _percentile(vals, robust_pct)
    hi = _percentile(vals, 100 - robust_pct)
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: min(1.0, max(0.0, (v - lo) / span)) for k, v in values.items()}


def load_block_values(path: Path, metric: str, bits: str) -> dict[str, float]:
    raw = json.loads(path.read_text())
    out: dict[str, float] = {}
    for block_name, block in raw.items():
        if metric not in block:
            raise KeyError(f"{path.name}: block {block_name!r} has no {metric!r} key.")
        by_bit = block[metric]
        if bits not in by_bit:
            available = ", ".join(sorted(by_bit))
            raise KeyError(f"{path.name}: {metric}[{bits!r}] not found for block {block_name!r} "
                            f"(available bit-widths: {available}).")
        v = float(by_bit[bits])
        if v <= 0:
            raise ValueError(f"{path.name}: block {block_name!r} has {metric}[{bits}] = {v} <= 0 -- "
                              "log-transform requires strictly positive values.")
        out[block_name] = v
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to a block_sensitivity_*.json file.")
    parser.add_argument("--metric", choices=["sensitivity_w", "sensitivity_a"], default="sensitivity_w",
                         help="Which per-block sensitivity series to plot (default: sensitivity_w).")
    parser.add_argument("--bits", default="4",
                         help="Candidate bit-width key to read, e.g. 2/4/8 (default: 4).")
    parser.add_argument("--robust-pct", type=float, default=5.0,
                         help="Percentile used for the robust panel's clip bounds, matching "
                              "ilp_search.py's --robust-normalize-pct (default: 5.0).")
    parser.add_argument("--highlight", nargs="+", default=None,
                         help="Block name(s) to call out in a distinct color (default: the 2 "
                              "blocks with the largest raw value).")
    parser.add_argument("--title", default=None,
                         help="Plot title context (default: derived from the input filename).")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output PNG path (default: compression/analysis/sensitivity/out/"
                              "<input_stem>_<metric>_<bits>bit.png).")
    return parser.parse_args()


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def make_figure(values: dict[str, float], metric: str, bits: str, robust_pct: float,
                 highlight: list[str], title: str):
    import matplotlib.pyplot as plt

    names_sorted = sorted(values, key=lambda k: values[k])  # ascending raw value
    n = len(names_sorted)
    raw = values
    log_values = {k: math.log10(v) for k, v in values.items()}
    log_mm = _normalize_1d(log_values, 0.0)
    mm = _normalize_1d(values, 0.0)
    rb = _normalize_1d(values, robust_pct)

    def color_for(name: str) -> str:
        return HIGHLIGHT_COLOR if name in highlight else OTHER_COLOR

    colors = [color_for(name) for name in names_sorted]
    y = list(range(n))

    fig, axes = plt.subplots(1, 4, figsize=(16, max(4.0, 0.28 * n + 1.4)), facecolor=SURFACE, sharey=True)
    fig.suptitle(title, color=INK, fontsize=13, y=0.995)

    # Panel 1: raw, log-x lollipop (barh can't start a log-scale bar at 0).
    ax = axes[0]
    _style_axes(ax)
    ax.set_xscale("log")
    xmin = min(raw.values()) * 0.6
    xmax = max(raw.values()) * 1.8
    for yi, name in zip(y, names_sorted):
        ax.hlines(yi, xmin, raw[name], color=GRID, linewidth=1.3, zorder=1)
        ax.plot(raw[name], yi, "o", color=color_for(name), markersize=6,
                 markeredgecolor=INK, markeredgewidth=0.4, zorder=3)
    ax.set_xlim(xmin, xmax)
    ax.set_title("raw", color=INK, fontsize=10.5)
    ax.set_xlabel(f"{metric} ({bits}-bit) -- log scale", color=SECONDARY_INK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=8, family="monospace", color=SECONDARY_INK)
    for label, name in zip(ax.get_yticklabels(), names_sorted):
        if name in highlight:
            label.set_color(HIGHLIGHT_COLOR)
            label.set_fontweight("bold")
    ax.invert_yaxis()  # smallest value at top, matching the ascending sort

    panels = [
        (axes[1], log_mm, "log → min–max", "log10, then min–max to [0,1]"),
        (axes[2], mm, "min–max", "(v − min) / (max − min)"),
        (axes[3], rb, f"robust {robust_pct:g}–{100 - robust_pct:g}",
         f"clipped to the {robust_pct:g}th/{100 - robust_pct:g}th percentile"),
    ]
    for ax, series, panel_title, xlabel in panels:
        _style_axes(ax)
        ax.barh(y, [series[name] for name in names_sorted], color=colors, height=0.62, zorder=3)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(panel_title, color=INK, fontsize=10.5)
        ax.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=8.5)
        ax.tick_params(labelleft=False)

    for name in highlight:
        yi = names_sorted.index(name)
        for ax, series in ((axes[1], log_mm), (axes[2], mm), (axes[3], rb)):
            ax.annotate(f"{series[name]:.3f}", (series[name], yi), color=INK, fontsize=7.5,
                        fontweight="bold", ha="left", va="center", xytext=(4, 0), textcoords="offset points")
        axes[0].annotate(f"{raw[name]:.2e}", (raw[name], names_sorted.index(name)), color=INK, fontsize=7.5,
                          fontweight="bold", ha="left", va="center", xytext=(6, 0), textcoords="offset points")

    highlight_label = "highlighted: " + ", ".join(highlight) if len(highlight) <= 4 else f"highlighted ({len(highlight)})"
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=OTHER_COLOR, markersize=8,
                   label=f"other blocks ({n - len(highlight)})"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=HIGHLIGHT_COLOR, markersize=8,
                   label=highlight_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=SECONDARY_INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig, names_sorted, log_mm, mm, rb


def print_diagnostic(names_sorted: list[str], values: dict[str, float],
                      log_mm: dict[str, float], mm: dict[str, float]) -> None:
    """The generalized version of the down1/regular1.1 finding: whichever
    space you min-max in, the two most extreme points in THAT space set
    the scale, and the "wasted" gap between the pack and the outlier can
    be comparably large on either end."""
    top, second = names_sorted[-1], names_sorted[-2]
    bottom, second_bottom = names_sorted[0], names_sorted[1]
    top_wasted = (1.0 - mm[second]) * 100
    bottom_wasted = log_mm[second_bottom] * 100
    print(f"\n{'block':<16}{'raw':>14}{'log->minmax':>14}{'min-max':>10}")
    for name in names_sorted:
        print(f"{name:<16}{values[name]:>14.4e}{log_mm[name]:>14.4f}{mm[name]:>10.4f}")
    print(f"\nLinear min-max: {top!r} sets the ceiling ({values[top]:.3e}); the next-highest, "
          f"{second!r} ({values[second]:.3e}, {values[top] / values[second]:.2f}x smaller), lands at "
          f"{mm[second]:.3f} -- {top_wasted:.1f}% of the [0,1] range is empty space above it.")
    print(f"Log -> min-max: {bottom!r} sets the floor ({values[bottom]:.3e}); the next-lowest, "
          f"{second_bottom!r} ({values[second_bottom]:.3e}, {values[second_bottom] / values[bottom]:.2f}x larger), "
          f"lands at {log_mm[second_bottom]:.3f} -- {bottom_wasted:.1f}% of the [0,1] range is empty space below it.")
    print("Log-transforming trades one lopsided tail for the other -- it does not, by itself, fix "
          "outlier fragility. Pair it with percentile clipping (--robust-pct) or a median/MAD scale "
          "to protect both tails at once.\n")


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"{args.input} not found.")
        return 1

    values = load_block_values(args.input, args.metric, args.bits)
    highlight = args.highlight
    if highlight is None:
        highlight = sorted(values, key=lambda k: values[k], reverse=True)[:2]
    else:
        missing = [name for name in highlight if name not in values]
        if missing:
            print(f"--highlight names not found in {args.input.name}: {missing}")
            return 1

    title = args.title or f"{args.input.stem} — {args.metric} @ {args.bits}-bit"
    fig, names_sorted, log_mm, mm, rb = make_figure(values, args.metric, args.bits, args.robust_pct,
                                                      highlight, title)

    out_path = args.out
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / f"{args.input.stem}_{args.metric}_{args.bits}bit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Wrote {out_path} ({len(values)} blocks)")

    print_diagnostic(names_sorted, values, log_mm, mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
