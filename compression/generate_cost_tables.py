"""Stage 2.2 cost tables (agent_instructions_1.yaml stage_2_architecture_grid.
2_2_cost_tables): architecture-only marginal cost of +delta filters / +1
bottleneck per stage, no training. Run once, upfront -- informs the grid
(2.3), the optional fine-tune (2.6), and the write-up.

Baseline = Original (16,64,128,64,16) -- re-baselined from E1 (20,72,144,72,20)
after real test-set results showed Original beats E1 on Dice (0.780 vs
0.775) while already being smaller (369k vs 466k params); see
foundation_log.md's "Plan revision 2" section. ENet-native bottleneck depth
(4,8,8,2,1), upsample_conv decoder (avoids the max_unpool channel-symmetry
constraint when perturbing a single channel slot in isolation -- see
ENet.py's self-test / foundation_log.md finding #2).

Note: ENet.py's `channels` tuple has 5 slots (initial, stage1, stage23,
stage4, stage5) -- stage2 and stage3 always share one width (no resolution
change between them, matching every row of the .md/yaml's f_i..f5 tables,
where f2 always equals f3). The filter-cost table below has 5 rows, not 6,
for that reason -- f2/f3 is one knob, not two.

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
BASELINE_CHANNELS = (16, 64, 128, 64, 16)  # Original -- re-baselined from E1, see foundation_log.md's "Plan revision 2" (real test-set Dice: Original 0.780 > E1 0.775, while already smaller)
BASELINE_BOTTLENECKS = (4, 8, 8, 2, 1)  # ENet-native
DECODER_TYPE = "upsample_conv"
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
        in_channels=1, out_channels=2, channels=channels,
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


# --- Activation / feature-map memory axis -----------------------------------
# A third axis distinct from params and FLOPs: how big is each stage's output
# feature map (elements/channel = H x W at that stage's resolution)? This
# doesn't depend on channel width (linear, not quadratic like params/flops --
# doubling channels exactly doubles memory, no block-count dependence either,
# since it's the OUTPUT size, not cumulative compute). Matters for on-chip
# buffering (BRAM) on the ZU7EV target, where activation memory is often the
# real constraint, not weight storage. Verified via forward hooks (real
# tensor shapes), not stride arithmetic, so it can't drift from the actual
# downsampling schedule if that ever changes.
ACTIVATION_HOOK_TARGETS = [
    ("f_i", "initial", "#2a78d6"),
    ("f1", "regular1", "#eb6834"),
    ("f2=f3 (stage23)", "stage3", "#1baf7a"),
    ("f4", "regular4", "#eda100"),
    ("f5", "regular5", "#e87ba4"),
]


def main_activation_memory() -> None:
    import torch

    OUT_DIR.mkdir(exist_ok=True)
    model = ENet(in_channels=1, out_channels=2, channels=BASELINE_CHANNELS,
                 bottlenecks_per_stage=BASELINE_BOTTLENECKS, decoder_type=DECODER_TYPE)
    shapes: dict[str, tuple[int, ...]] = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            shapes[name] = tuple(tensor.shape)
        return hook

    handles = [getattr(model, attr).register_forward_hook(make_hook(name))
               for name, attr, _ in ACTIVATION_HOOK_TARGETS]
    with torch.no_grad():
        model(torch.zeros(1, 1, *INPUT_HW))
    for handle in handles:
        handle.remove()

    rows = []
    for name, attr, color in ACTIVATION_HOOK_TARGETS:
        _, channels_at_baseline, h, w = shapes[name]
        rows.append({
            "stage": name,
            "resolution": f"{h}x{w}",
            "elements_per_channel": h * w,
            "note": f"channels-independent -- total activation memory for this stage = "
                    f"elements_per_channel x channel_width x bytes/element (bytes/element "
                    f"depends on Stage 4's quant_bits; 4 for FP32).",
        })
    activation_df = pd.DataFrame(rows)
    activation_df.to_csv(OUT_DIR / "activation_memory.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'activation_memory.csv'}")
    print(activation_df.to_string(index=False))

    import matplotlib.pyplot as plt
    ink, secondary_ink, muted, grid, baseline, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(True, axis="y", color=grid, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(baseline)
    ax.tick_params(colors=muted, labelsize=9)
    names = [name for name, _, _ in ACTIVATION_HOOK_TARGETS]
    colors = [color for _, _, color in ACTIVATION_HOOK_TARGETS]
    values = [activation_df.loc[activation_df["stage"] == name, "elements_per_channel"].iloc[0] for name in names]
    ax.bar(names, values, color=colors, zorder=3, width=0.6)
    ax.set_ylabel("elements per channel (H x W at that stage)", color=secondary_ink, fontsize=10)
    ax.set_title(f"Feature-map size per channel by stage (input {INPUT_HW[0]}x{INPUT_HW[1]})", color=ink, fontsize=12)
    for i, v in enumerate(values):
        ax.annotate(f"{v:,}", (i, v), ha="center", va="bottom", fontsize=9, color=secondary_ink, xytext=(0, 4), textcoords="offset points")
    fig.tight_layout()
    out_path = OUT_DIR / "activation_memory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=surface)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
    main_normalized()
    main_activation_memory()
