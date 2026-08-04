"""Config-independent cost model: params-per-added-filter/-block AND
flops-per-added-filter/-block as CONTINUOUS functions of current channel
width, per stage role -- generalizes beyond the specific grid points (E1,
U2..UF), unlike cost_tables/*.csv's point measurements.

Why this is the right decomposition (see generate_cost_tables.py's
normalized_* tables for the two-point version this replaces with a full
curve): a stage's total cost ~= N_blocks x f(width), and N_blocks scales
in exactly linearly (adding a block never changes any other block's cost).
So the whole joint (width, depth) surface collapses to one curve per stage
-- f'(width) (this script's per-block panels) -- plus a linear multiply by N:

    cost of widening a stage by delta-C, at depth N
        ~= N x [f(C + delta_C) - f(C)]
        ~= N x delta_C x f'(C)   (for small delta_C)

...for either params or FLOPs -- same shape of relationship, different
currency (agent_instructions_1.yaml's efficiency_fp32: [params, flops]).

Figure 1 (cost_relationships.png): params, per-filter + per-block.
Figure 2 (cost_relationships_flops.png): FLOPs, per-filter + per-block, at
512x512 (matches cost_tables/filter_cost.csv's resolution).

Usage: python compression/plot_cost_relationships.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_flops, count_params  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "cost_tables"
FLOOR_CHANNELS = (4, 4, 4, 4, 4)
FLOOR_BOTTLENECKS = (1, 1, 1, 1, 1)
C_RANGE = list(range(4, 164, 4))  # covers below-UF up to beyond E1's widest (144)
FILTER_DELTA = 4
INPUT_HW = (512, 512)
OUT_CHANNELS = 5  # background + LAD/RCA/LCX/LM, matches Dataset509_ARCADE_1x1_4c's dataset.json

# Palette: dataviz skill's validated categorical order (fixed order, not cycled).
# channel_idx indexes ENet's `channels` tuple (initial, stage1, stage23, stage4, stage5);
# bottleneck_idx indexes `bottlenecks_per_stage` (n_stage1, n_stage2, n_stage3, n_regular4,
# n_regular5) -- these are DIFFERENT tuples with different semantics, not the same index
# space (e.g. regular1 -- bottlenecks_per_stage[0] -- operates at stage1_channels, which is
# channels[1], not channels[0]). f_i has no bottleneck-depth counterpart (InitialBlock is
# never repeated) -- bottleneck_idx=None, block panel skipped for it.
SERIES = [
    ("f_i", 0, None, "#2a78d6"),              # slot 1 blue
    ("f1", 1, 0, "#eb6834"),                  # slot 2 orange -- regular1
    ("f2=f3 (stage23)", 2, 2, "#1baf7a"),     # slot 3 aqua -- stage3 (stage2 idx=1 would be equivalent, same width)
    ("f4", 3, 3, "#eda100"),                  # slot 4 yellow -- regular4
    ("f5", 4, 4, "#e87ba4"),                  # slot 5 magenta -- regular5
]
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def metrics_at(channel_idx: int, width: int, bottleneck_idx: int | None, bottleneck_count: int) -> tuple[int, float]:
    channels = list(FLOOR_CHANNELS)
    channels[channel_idx] = width
    bottlenecks = list(FLOOR_BOTTLENECKS)
    if bottleneck_idx is not None:
        bottlenecks[bottleneck_idx] = bottleneck_count
    model = ENet(in_channels=1, out_channels=OUT_CHANNELS, channels=tuple(channels),
                 bottlenecks_per_stage=tuple(bottlenecks), decoder_type="upsample_conv")
    params, _ = count_params(model)
    _, flops = count_flops(model, 1, INPUT_HW)
    return params, flops


def build_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_filter_rows = []
    per_block_rows = []
    for name, channel_idx, bottleneck_idx, color in SERIES:
        for width in C_RANGE:
            p_here, f_here = metrics_at(channel_idx, width, bottleneck_idx, 1)
            p_next, f_next = metrics_at(channel_idx, width + FILTER_DELTA, bottleneck_idx, 1)
            per_filter_rows.append({
                "stage": name, "width": width,
                "params_per_filter": (p_next - p_here) / FILTER_DELTA,
                "flops_per_filter": (f_next - f_here) / FILTER_DELTA,
            })

            if bottleneck_idx is None:
                continue  # f_i: no repeated-block concept, block panel doesn't apply
            p_one, f_one = metrics_at(channel_idx, width, bottleneck_idx, 1)
            p_two, f_two = metrics_at(channel_idx, width, bottleneck_idx, 2)
            per_block_rows.append({
                "stage": name, "width": width,
                "params_per_block": p_two - p_one,
                "flops_per_block": f_two - f_one,
            })
    return pd.DataFrame(per_filter_rows), pd.DataFrame(per_block_rows)


def styled_axes(*axes) -> None:
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)


def plot_pair(per_filter_df: pd.DataFrame, per_block_df: pd.DataFrame, metric: str, unit: str, title_suffix: str, out_name: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    styled_axes(ax1, ax2)

    filter_col = f"{metric}_per_filter"
    block_col = f"{metric}_per_block"

    for name, channel_idx, bottleneck_idx, color in SERIES:
        sub = per_filter_df[per_filter_df["stage"] == name]
        ax1.plot(sub["width"], sub[filter_col], color=color, linewidth=2, label=name, zorder=3)
    ax1.set_xlabel("current channel width (C)", color=SECONDARY_INK, fontsize=10)
    ax1.set_ylabel(f"{unit} per +1 filter", color=SECONDARY_INK, fontsize=10)
    ax1.set_title(f"Marginal {title_suffix} per added filter\n(all stages, incl. f_i for reference)", color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)

    for name, channel_idx, bottleneck_idx, color in SERIES:
        if bottleneck_idx is None:
            continue
        sub = per_block_df[per_block_df["stage"] == name]
        ax2.plot(sub["width"], sub[block_col], color=color, linewidth=2, label=name, zorder=3)
    ax2.set_xlabel("current channel width (C)", color=SECONDARY_INK, fontsize=10)
    ax2.set_ylabel(f"{unit} per +1 bottleneck block", color=SECONDARY_INK, fontsize=10)
    ax2.set_title(f"Marginal {title_suffix} per added block\n(exact + linear in block count, at fixed width)", color=INK, fontsize=11)
    ax2.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)

    fig.suptitle(f"ENet stage cost curves ({title_suffix}) — config-independent (holds for any grid point)",
                 color=INK, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    per_filter_df, per_block_df = build_curves()
    per_filter_df.to_csv(OUT_DIR / "cost_curve_per_filter.csv", index=False)
    per_block_df.to_csv(OUT_DIR / "cost_curve_per_block.csv", index=False)
    print(f"Wrote {OUT_DIR / 'cost_curve_per_filter.csv'} and cost_curve_per_block.csv (params + flops columns)")

    plot_pair(per_filter_df, per_block_df, metric="params", unit="params", title_suffix="params", out_name="cost_relationships.png")
    plot_pair(per_filter_df, per_block_df, metric="flops", unit="FLOPs", title_suffix="FLOPs", out_name="cost_relationships_flops.png")

    # Worked example, using the fitted curves rather than a re-measurement --
    # demonstrates the "N x per-filter-rate" joint relationship concretely,
    # for both currencies.
    example_stage, example_channel_idx, example_bottleneck_idx, _ = SERIES[2]  # stage23 -- the most expensive slot
    example_width, example_delta, example_n = 20, 4, 3
    p_here, f_here = metrics_at(example_channel_idx, example_width, example_bottleneck_idx, 1)
    p_next, f_next = metrics_at(example_channel_idx, example_width + example_delta, example_bottleneck_idx, 1)
    params_rate, flops_rate = p_next - p_here, f_next - f_here
    print(f"\nWorked example: widening {example_stage} from {example_width} by +{example_delta} channels, "
          f"at {example_n} bottleneck blocks:")
    print(f"  per-block delta at width {example_width}->{example_width+example_delta}: {params_rate} params, {flops_rate:.0f} flops")
    print(f"  x {example_n} blocks = {example_n*params_rate} params, {example_n*flops_rate:.0f} flops total "
          "(before any other stage's connector changes)")
    print("  (channel counts must stay multiples of 4 in the real model -- +5 isn't buildable, +4/+8 are)")


if __name__ == "__main__":
    main()
