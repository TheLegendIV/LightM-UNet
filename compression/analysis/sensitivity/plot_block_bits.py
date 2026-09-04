"""Plots a block_bits_*.json (ilp_search.py's own output -- {'stage_weight_bits':
{...}, 'stage_act_bits': {...}, '_diagnostics': {...}}) as the per-block
bit-width PROFILE across network depth: one point per block per axis
(weight/activation), in the network's real forward-pass order, so the
SHAPE of a quantization decision is visible at a glance -- which stretches
of the network got pushed to 8-bit, whether that's clustered or scattered,
and (via --fix-bits's own recorded `_diagnostics.fixed_bits`) which of
those were actually decided by the ILP versus pinned by hand.

This is the companion to plot_block_normalization.py (which plots the
INPUT -- raw sensitivity -- before the ILP runs); this one plots the
OUTPUT -- the bit-width the ILP (or --fix-bits) actually chose.

Usage:
    python compression/analysis/sensitivity/plot_block_bits.py \\
        compression/hawq/artifacts/block_bits_12_separable_dense_relu_min4_fixr5_4_down1_8.json

    # compare against a baseline (e.g. the unfixed run) -- baseline value
    # shown as a faint ghost marker, changed blocks get a ring around them:
    python compression/analysis/sensitivity/plot_block_bits.py \\
        compression/hawq/artifacts/block_bits_12_separable_dense_relu_min4_fixr5_4_down1_8.json \\
        --baseline compression/hawq/artifacts/block_bits_12_separable_dense_relu_min4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"
# S12's own block_sensitivity file preserves the network's real depth order
# (initial -> down1 -> regular1.* -> down2 -> stage2.* -> stage3.* -> up4 ->
# regular4.* -> up5 -> regular5.0 -> final) -- block_bits_*.json's own key
# order is NOT reliable for this once --fix-bits has been used (pinned
# blocks get appended at the end, out of depth order -- see ilp_search.py's
# main()), so ordering is sourced from a separate reference file instead.
DEFAULT_ORDER_FROM = REPO_ROOT / "compression/hawq/artifacts/block_sensitivity_12_separable_dense_relu.json"

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
FREE_COLOR = "#2a78d6"    # slot 1 blue -- ILP-decided
FIXED_COLOR = "#eb6834"   # slot 2 orange -- pinned via --fix-bits
CHANGED_RING = "#0b0b0b"  # ring around a marker that differs from --baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to a block_bits_*.json (ilp_search.py output).")
    parser.add_argument("--order-from", type=Path, default=DEFAULT_ORDER_FROM,
                         help="A JSON file whose top-level key order defines the x-axis's network-depth "
                              "order (default: S12's own block_sensitivity file). Only its key ORDER is "
                              "used, not its values.")
    parser.add_argument("--baseline", type=Path, default=None,
                         help="Another block_bits_*.json to diff against -- blocks whose bit-width "
                              "differs get a black ring, plus a faint ghost marker at the baseline value.")
    parser.add_argument("--title", default=None, help="Plot title (default: derived from the input filename).")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output PNG path (default: compression/analysis/sensitivity/out/<input_stem>_bits.png).")
    return parser.parse_args()


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _ordered_names(bits_dict: dict[str, int], order_from: Path) -> list[str]:
    if order_from.exists():
        ref_order = list(json.loads(order_from.read_text()).keys())
        ordered = [n for n in ref_order if n in bits_dict]
        leftover = [n for n in bits_dict if n not in ordered]
        if leftover:
            print(f"Note: {leftover} not found in {order_from.name}'s own key order -- appended at the end.")
        return ordered + leftover
    print(f"Note: --order-from {order_from} not found -- falling back to {list(bits_dict)!r}'s own file order.")
    return list(bits_dict)


def _group_boundaries(names: list[str]) -> list[int]:
    """Index positions where the block's group prefix (text before the
    first '.', or the whole name for un-dotted blocks like initial/down1/
    up4/final) changes from the previous block -- drawn as light vertical
    dividers so the stage structure reads at a glance."""
    def prefix(n: str) -> str:
        return n.split(".")[0]
    return [i for i in range(1, len(names)) if prefix(names[i]) != prefix(names[i - 1])]


def make_figure(data: dict, baseline: dict | None, order_from: Path, title: str):
    import matplotlib.pyplot as plt

    weight_bits = data["stage_weight_bits"]
    act_bits = data["stage_act_bits"]
    fixed = set(data.get("_diagnostics", {}).get("fixed_bits", {}))
    names = _ordered_names(weight_bits, order_from)
    x = list(range(len(names)))
    boundaries = _group_boundaries(names)

    base_w = baseline["stage_weight_bits"] if baseline else None
    base_a = baseline["stage_act_bits"] if baseline else None

    all_bits = list(weight_bits.values()) + list(act_bits.values())
    if base_w:
        all_bits += list(base_w.values()) + list(base_a.values())
    y_ticks = sorted(set(all_bits))

    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.34 * len(names)), 6.5), facecolor=SURFACE, sharex=True)
    fig.suptitle(title, color=INK, fontsize=13, y=0.99)

    panels = [(axes[0], weight_bits, base_w, "weight bits"), (axes[1], act_bits, base_a, "activation bits")]
    for ax, series, base_series, ylabel in panels:
        _style_axes(ax)
        for b in boundaries:
            ax.axvline(b - 0.5, color=GRID, linewidth=0.9, zorder=1)
        vals = [series[n] for n in names]
        colors = [FIXED_COLOR if n in fixed else FREE_COLOR for n in names]
        ax.plot(x, vals, color=MUTED, linewidth=1.1, zorder=2)
        if base_series:
            base_vals = [base_series[n] for n in names]
            changed = [n for n in names if base_series[n] != series[n]]
            ax.scatter(x, base_vals, marker="_", s=90, color=MUTED, alpha=0.55, zorder=2,
                       label="baseline" if ax is axes[0] else None)
            ring_x = [names.index(n) for n in changed]
            ring_y = [series[n] for n in changed]
            ax.scatter(ring_x, ring_y, s=170, facecolors="none", edgecolors=CHANGED_RING,
                       linewidths=1.3, zorder=4)
        ax.scatter(x, vals, color=colors, s=70, zorder=5, edgecolors=INK, linewidths=0.5)
        ax.set_yticks(y_ticks)
        ax.set_ylim(min(y_ticks) - 1, max(y_ticks) + 1)
        ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=90, fontsize=7.5, family="monospace", color=SECONDARY_INK)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=FREE_COLOR, markersize=8, markeredgecolor=INK,
                   label=f"ILP-decided ({len(names) - len(fixed)})"),
    ]
    if fixed:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=FIXED_COLOR, markersize=8,
                                   markeredgecolor=INK, label=f"fixed via --fix-bits ({len(fixed)})"))
    if baseline:
        handles.append(plt.Line2D([0], [0], marker="_", linestyle="", color=MUTED, markersize=12,
                                   label="baseline value"))
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                                   markeredgecolor=CHANGED_RING, markersize=11, label="changed from baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=SECONDARY_INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def _summary_line(data: dict) -> str:
    w = list(data["stage_weight_bits"].values())
    a = list(data["stage_act_bits"].values())
    diag = data.get("_diagnostics", {})
    avg_w, avg_a = sum(w) / len(w), sum(a) / len(a)
    parts = [f"avg_w={avg_w:.3f}", f"avg_a={avg_a:.3f}", f"avg={((avg_w + avg_a) / 2):.3f}"]
    if "lut_pct_of_budget" in diag:
        parts.append(f"LUT={diag['lut_pct_of_budget']:.1f}%")
    if "bram_pct_of_budget" in diag:
        parts.append(f"BRAM={diag['bram_pct_of_budget']:.1f}%")
    fixed = diag.get("fixed_bits")
    if fixed:
        parts.append(f"fixed={fixed}")
    return ", ".join(parts)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"{args.input} not found.")
        return 1
    data = json.loads(args.input.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None

    title = args.title or f"{args.input.stem}\n{_summary_line(data)}"
    fig = make_figure(data, baseline, args.order_from, title)

    out_path = args.out or DEFAULT_OUT_DIR / f"{args.input.stem}_bits.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(_summary_line(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
