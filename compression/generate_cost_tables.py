"""Architecture-only marginal cost of +delta filters / +1 bottleneck per
stage, no training -- informs stage_1_naive_baseline's grid and any further
architecture probes, for the current 4-class objective (LAD/RCA/LCX/LM on
Dataset509_ARCADE_1x1_4c).

Baseline = ENet-paper channels (16,64,128,64,16) -- same as stage_1's own
"Baseline" config. ENet-native bottleneck depth (4,8,8,2,1), upsample_conv
decoder (avoids the max_unpool channel-symmetry constraint when perturbing a
single channel slot in isolation).

Note: ENet.py's `channels` tuple has 5 slots (initial, stage1, stage23,
stage4, stage5) -- stage2 and stage3 always share one width (no resolution
change between them). The filter-cost table below has 5 rows, not 6, for
that reason -- f2/f3 is one knob, not two.

Only parameter count and MACs/FLOPs are reported (this session dropped the
activation-memory axis this file used to also produce -- see rank_results.py
for the current cost-vs-accuracy scoring, which only uses params/MACs/Dice).

Usage: python compression/generate_cost_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_flops, count_params  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "cost_tables"
BASELINE_CHANNELS = (16, 64, 128, 64, 16)  # ENet-paper channels -- same as stage_1_naive_baseline's "Baseline" config
BASELINE_BOTTLENECKS = (4, 8, 8, 2, 1)  # ENet-native
DECODER_TYPE = "upsample_conv"
OUT_CHANNELS = 5  # background + LAD/RCA/LCX/LM, matches Dataset509_ARCADE_1x1_4c's dataset.json
INPUT_HW = (512, 512)
FILTER_DELTA = 4  # smallest valid step (stage channels must stay divisible by 4)

FILTER_SLOTS = [
    ("f_i", 0, "initial block width; feeds InitialBlock + down1's main branch only"),
    ("f1", 1, "stage1 (post-down1), 1/4 resolution"),
    ("f2=f3 (stage23)", 2, "shared context-stage width, 1/8 resolution (both stage2 and stage3 run here)"),
    ("f4", 3, "regular4 (post-up4), 1/4 resolution"),
    ("f5", 4, "stage5 (post-up5), 1/2 resolution"),
]
BOTTLENECK_SLOTS = [
    ("s1", 0, "regular1, 1/4 resolution"),
    ("s2", 1, "stage2 context, 1/8 resolution"),
    ("s3", 2, "stage3 context, 1/8 resolution -- same resolution as s2, so same marginal cost"),
    ("s4", 3, "regular4, 1/4 resolution"),
    ("s5", 4, "regular5, 1/2 resolution"),
]

# --- Normalized table -------------------------------------------------------
# filter_cost/bottleneck_cost above are each anchored at the OTHER axis's E1/
# native value -- not independent. Depth (bottleneck count) genuinely IS
# linear/additive (N identical blocks cost exactly N x one block, at whatever
# width), so a per-block-at-a-reference-width number is exact and reusable.
# Width is NOT linear -- a bottleneck block's params are dominated by its 1x1
# reduce/expand convs, each ~proportional to channels, so total params grow
# ~quadratically with width. A single "per filter" rate is therefore only
# ever a local slope, never a universal constant -- moving the anchor from
# E1 (wide) to a floor width doesn't remove that, it just relocates it. This
# table reports the floor-width anchor AND a second, wider sample per slot so
# the non-constant rate is visible in the data, not asserted in a comment.
FLOOR_CHANNELS = (4, 4, 4, 4, 4)
FLOOR_BOTTLENECKS = (1, 1, 1, 1, 1)
WIDE_SAMPLE_CHANNEL = 20  # second width sample, per filter slot: floor(4) -> WIDE_SAMPLE_CHANNEL


def main_normalized() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Per-bottleneck cost at floor width (4): exact and linear at this width
    # -- N blocks at width 4 cost exactly N x this number, for any N.
    floor_base_params, floor_base_flops = build(FLOOR_CHANNELS, FLOOR_BOTTLENECKS)
    bottleneck_rows = []
    for name, idx, _ in BOTTLENECK_SLOTS:
        bottlenecks = list(FLOOR_BOTTLENECKS)
        bottlenecks[idx] += 1
        params, flops = build(FLOOR_CHANNELS, tuple(bottlenecks))
        bottleneck_rows.append({
            "stage": name,
            "params_per_bottleneck_at_width4": params - floor_base_params,
            "flops_per_bottleneck_at_width4": flops - floor_base_flops,
            "note": "exact + linear at width=4 -- N blocks costs exactly N x this, at width=4. "
                    "Rescale for other widths using the quadratic width relationship below, not linearly.",
        })

    # Per-filter cost at TWO width anchors (floor=4 and WIDE_SAMPLE_CHANNEL),
    # both at bottleneck depth=1 -- shows the rate is not constant.
    filter_rows = []
    for name, idx, _ in FILTER_SLOTS:
        channels_floor = list(FLOOR_CHANNELS)
        channels_floor[idx] += FILTER_DELTA
        params_floor, flops_floor = build(tuple(channels_floor), FLOOR_BOTTLENECKS)
        rate_floor_params = (params_floor - floor_base_params) / FILTER_DELTA
        rate_floor_flops = (flops_floor - floor_base_flops) / FILTER_DELTA

        wide_base_channels = list(FLOOR_CHANNELS)
        wide_base_channels[idx] = WIDE_SAMPLE_CHANNEL
        wide_base_params, wide_base_flops = build(tuple(wide_base_channels), FLOOR_BOTTLENECKS)
        wide_channels = list(wide_base_channels)
        wide_channels[idx] += FILTER_DELTA
        params_wide, flops_wide = build(tuple(wide_channels), FLOOR_BOTTLENECKS)
        rate_wide_params = (params_wide - wide_base_params) / FILTER_DELTA
        rate_wide_flops = (flops_wide - wide_base_flops) / FILTER_DELTA

        filter_rows.append({
            "stage": name,
            "params_per_filter_at_width4": round(rate_floor_params, 2),
            f"params_per_filter_at_width{WIDE_SAMPLE_CHANNEL}": round(rate_wide_params, 2),
            "rate_ratio": round(rate_wide_params / rate_floor_params, 2) if rate_floor_params else None,
            "flops_per_filter_at_width4": round(rate_floor_flops, 1),
            f"flops_per_filter_at_width{WIDE_SAMPLE_CHANNEL}": round(rate_wide_flops, 1),
        })

    bottleneck_df = pd.DataFrame(bottleneck_rows)
    bottleneck_df.to_csv(OUT_DIR / "normalized_bottleneck_cost.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'normalized_bottleneck_cost.csv'}")
    print(bottleneck_df.to_string(index=False))

    filter_df = pd.DataFrame(filter_rows)
    filter_df.to_csv(OUT_DIR / "normalized_filter_cost.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'normalized_filter_cost.csv'}")
    print(filter_df.to_string(index=False))


def build(channels: tuple[int, ...], bottlenecks: tuple[int, ...]) -> tuple[int, float]:
    model = ENet(
        in_channels=1, out_channels=OUT_CHANNELS, channels=channels,
        bottlenecks_per_stage=bottlenecks, decoder_type=DECODER_TYPE,
    )
    params, _ = count_params(model)
    _, flops = count_flops(model, 1, INPUT_HW)
    return params, flops


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    base_params, base_flops = build(BASELINE_CHANNELS, BASELINE_BOTTLENECKS)
    print(f"Baseline (Original, native bottlenecks, {DECODER_TYPE}): params={base_params} flops={base_flops:.0f}")

    filter_rows = []
    for name, idx, note in FILTER_SLOTS:
        channels = list(BASELINE_CHANNELS)
        channels[idx] += FILTER_DELTA
        params, flops = build(tuple(channels), BASELINE_BOTTLENECKS)
        filter_rows.append({
            "stage": name, "delta_filters": FILTER_DELTA,
            "delta_params": params - base_params, "delta_flops": flops - base_flops,
            "notes": note,
        })
    filter_cost = pd.DataFrame(filter_rows)
    filter_cost.to_csv(OUT_DIR / "filter_cost.csv", index=False)
    print(f"Wrote {OUT_DIR / 'filter_cost.csv'}")
    print(filter_cost.to_string(index=False))

    bottleneck_rows = []
    for name, idx, note in BOTTLENECK_SLOTS:
        bottlenecks = list(BASELINE_BOTTLENECKS)
        bottlenecks[idx] += 1
        params, flops = build(BASELINE_CHANNELS, tuple(bottlenecks))
        bottleneck_rows.append({
            "stage": name, "delta_bottlenecks": 1,
            "delta_params": params - base_params, "delta_flops": flops - base_flops,
            "notes": note,
        })
    bottleneck_cost = pd.DataFrame(bottleneck_rows)
    bottleneck_cost.to_csv(OUT_DIR / "bottleneck_cost.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'bottleneck_cost.csv'}")
    print(bottleneck_cost.to_string(index=False))


if __name__ == "__main__":
    main()
    main_normalized()
