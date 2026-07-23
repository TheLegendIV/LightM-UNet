"""Hardware-savings ranking for the section 2a pruning grid, vs. the
Original baseline.

score = alpha * (MACs / MACs_Original)
      + beta  * ((sum(activation_elems x bits) + sum(params x bits)) / same_for_Original)
      - c     * (Dice / Dice_Original)

A MINIMIZATION function: lower score = cheaper hardware footprint relative
to Original, weighted against not losing accuracy (Dice term is SUBTRACTED,
not added, so higher relative Dice pulls the score down/better -- as
literally written with a plain "+" it would reward worse accuracy, which
contradicts "minimize" for a cost-benefit tradeoff; confirmed with the user
before computing anything). alpha=beta=c=1/3 (equal weighting) unless
overridden.

Re-baselined from E1 to Original this session: real test-set inference
showed Original beats E1 on Dice (0.7800 vs 0.7746) while already being
smaller (369,497 vs 466,294 params) -- see foundation_log.md's "Plan
revision 2" section. The filter axis (was E1/U2/U4/U8/U16/UF) is now
Original/O2/O4/O8/O16/OF, re-derived from Original's (16,64,128,64,16)
proportions rather than E1's.

The `bits` multiplier is a fixed assumed value (ASSUMED_BITS, default 8 --
the realistic Stage 4 target) applied uniformly to every config INCLUDING
Original, so it cancels exactly in the memory ratio: (act_elems_c + params_c)
x B / (act_elems_Original + params_Original) x B = (act_elems_c + params_c)
/ (act_elems_Original + params_Original), independent of B. It's kept in
the computation (not dropped) for transparency/documentation, and because
absolute (non-ratio) columns are also reported.

Dice: read from compression/results.csv (stage=2a_pruning_grid) if that
config has actually been trained; the Original baseline itself uses the
REAL measured value (0.7800019869577919, stage=stage1) rather than a
placeholder, now that it exists -- not a guess like the original 0.83
placeholder was. Grid cells not yet trained fall back to
DICE_ORIGINAL_PLACEHOLDER (i.e. dice_ratio=1, "assume parity with Original
until real training data exists"), flagged per-row via
`dice_is_placeholder`. Re-run this script after 2a actually trains and it
picks up real Dice automatically, no changes needed.

Activation-memory proxy uses the 5 characteristic per-stage
elements-per-channel constants already verified via forward hooks in
generate_cost_tables.py's main_activation_memory() (channel-width-
independent -- ENet's downsampling schedule is fixed by stride/pooling
ops, not channel counts, so these hold for every grid config, not just
Original) -- stage2 and stage3 share one width and are counted once
(matches activation_memory.csv's convention: this undercounts true total
activation traffic since both are real, separate feature maps, but does so
uniformly across every config, so the RANKING is unaffected, only absolute
values would be if reported outside a ratio).

Usage: python compression/generate_hardware_savings_ranking.py
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
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

ALPHA, BETA, C_WEIGHT = 1 / 3, 1 / 3, 1 / 3
ASSUMED_BITS = 8  # cancels in every ratio (see docstring) -- kept for transparency only
DICE_ORIGINAL_PLACEHOLDER = 0.7800019869577919  # REAL measured value (stage1, checkpoint_best.pth), not a guess

ORIGINAL_CHANNELS = (16, 64, 128, 64, 16)
ORIGINAL_BOTTLENECKS = (4, 8, 8, 2, 1)
DECODER_TYPE = "upsample_conv"  # matches pruning_2a_grid_array.job's forced choice (see foundation_log.md finding #2)
INPUT_HW = (512, 512)

# Channel-width-independent (resolution is fixed by stride/pooling, not
# channel count) -- verified via forward hooks in generate_cost_tables.py.
STAGE_ELEMENTS_PER_CHANNEL = {
    "f_i": 65536, "f1": 16384, "stage23": 4096, "f4": 16384, "f5": 65536,
}

FILTER_NAMES = ["Original", "O2", "O4", "O8", "O16", "OF"]
FILTER_CHANNELS = {
    "Original": (16, 64, 128, 64, 16), "O2": (16, 32, 64, 32, 8), "O4": (16, 16, 32, 16, 4),
    "O8": (16, 8, 16, 8, 4), "O16": (16, 4, 8, 4, 4), "OF": (16, 4, 4, 4, 4),
}
BOTTLENECK_NAMES = ["native", "5", "3", "2"]
BOTTLENECK_VALUES = {
    "native": (4, 8, 8, 2, 1), "5": (4, 5, 5, 2, 1), "3": (4, 3, 3, 2, 1), "2": (4, 2, 2, 2, 1),
}

# Categorical colors (dataviz skill's validated order), one per filter family.
FILTER_COLORS = {
    "Original": "#2a78d6", "O2": "#eb6834", "O4": "#1baf7a",
    "O8": "#eda100", "O16": "#e87ba4", "OF": "#008300",
}


def activation_elements(channels: tuple[int, ...]) -> int:
    f_i, f1, stage23, f4, f5 = channels
    return (
        STAGE_ELEMENTS_PER_CHANNEL["f_i"] * f_i
        + STAGE_ELEMENTS_PER_CHANNEL["f1"] * f1
        + STAGE_ELEMENTS_PER_CHANNEL["stage23"] * stage23
        + STAGE_ELEMENTS_PER_CHANNEL["f4"] * f4
        + STAGE_ELEMENTS_PER_CHANNEL["f5"] * f5
    )


def measure(channels: tuple[int, ...], bottlenecks: tuple[int, ...]) -> dict:
    model = ENet(in_channels=1, out_channels=2, channels=channels,
                 bottlenecks_per_stage=bottlenecks, decoder_type=DECODER_TYPE)
    params, _ = count_params(model)
    macs, _ = count_flops(model, 1, INPUT_HW)
    act_elems = activation_elements(channels)
    memory_bits = (act_elems + params) * ASSUMED_BITS
    return {"params": params, "macs": macs, "activation_elements": act_elems, "memory_bits": memory_bits}


def load_dice(config_name: str, existing_results: pd.DataFrame | None) -> tuple[float, bool]:
    if existing_results is not None:
        match = existing_results[existing_results["config_name"] == config_name]
        if not match.empty and pd.notna(match.iloc[0]["dice"]):
            return float(match.iloc[0]["dice"]), False
    return DICE_ORIGINAL_PLACEHOLDER, True


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    existing_results = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else None

    original = measure(ORIGINAL_CHANNELS, ORIGINAL_BOTTLENECKS)
    original_dice, original_dice_is_placeholder = load_dice("nnUNetTrainerENet_Original", existing_results)
    print(f"Original baseline: params={original['params']} macs={original['macs']:.3e} "
          f"activation_elements={original['activation_elements']} memory_bits={original['memory_bits']:.3e} "
          f"dice={original_dice} (placeholder={original_dice_is_placeholder})")

    rows = []
    for filter_name in FILTER_NAMES:
        for bneck_name in BOTTLENECK_NAMES:
            channels = FILTER_CHANNELS[filter_name]
            bottlenecks = BOTTLENECK_VALUES[bneck_name]
            config_name = f"2a_{filter_name}_{bneck_name}"
            m = measure(channels, bottlenecks)
            dice, dice_is_placeholder = load_dice(f"nnUNetTrainerENet_{config_name}", existing_results)

            macs_ratio = m["macs"] / original["macs"]
            mem_ratio = m["memory_bits"] / original["memory_bits"]
            dice_ratio = dice / original_dice
            score = ALPHA * macs_ratio + BETA * mem_ratio - C_WEIGHT * dice_ratio

            rows.append({
                "config_name": config_name, "filter": filter_name, "bottleneck": bneck_name,
                "params": m["params"], "macs": m["macs"], "activation_elements": m["activation_elements"],
                "memory_bits": m["memory_bits"],
                "macs_ratio": macs_ratio, "memory_ratio": mem_ratio,
                "dice": dice, "dice_ratio": dice_ratio, "dice_is_placeholder": dice_is_placeholder,
                "score": score,
            })

    df = pd.DataFrame(rows).sort_values("score", ignore_index=True)
    out_csv = OUT_DIR / "hardware_savings_ranking.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} configs, alpha={ALPHA:.3f} beta={BETA:.3f} c={C_WEIGHT:.3f})")
    print(df[["config_name", "macs_ratio", "memory_ratio", "dice_ratio", "dice_is_placeholder", "score"]].to_string(index=False))
    if df["dice_is_placeholder"].any():
        print(f"\nNOTE: {df['dice_is_placeholder'].sum()}/{len(df)} rows use the Dice PLACEHOLDER "
              f"({DICE_ORIGINAL_PLACEHOLDER}, assumed parity with Original) -- section 2a hasn't trained "
              "those cells yet. Re-run this script after it does; real Dice will be picked up automatically.")

    plot_ranking(df)


def plot_ranking(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ink, secondary_ink, muted, grid, baseline, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(True, axis="x", color=grid, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(baseline)
    ax.tick_params(colors=muted, labelsize=8)

    labels = [f"{row.filter}/bn{row.bottleneck}" for row in df.itertuples()]
    colors = [FILTER_COLORS[row.filter] for row in df.itertuples()]
    y_pos = range(len(df))
    ax.barh(y_pos, df["score"], color=colors, zorder=3, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # best (lowest score) at top
    ax.axvline(0, color=baseline, linewidth=1)

    placeholder_note = " (dice placeholder for untrained cells)" if df["dice_is_placeholder"].any() else ""
    ax.set_xlabel(f"score = {ALPHA:.2f}·MACs_ratio + {BETA:.2f}·memory_ratio − {C_WEIGHT:.2f}·Dice_ratio "
                  "(lower = more hardware savings vs. Original)", color=secondary_ink, fontsize=9)
    ax.set_title(f"Section 2a grid: hardware-savings ranking vs. Original{placeholder_note}", color=ink, fontsize=12)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FILTER_COLORS[name], label=name) for name in FILTER_NAMES]
    ax.legend(handles=handles, title="filter axis", frameon=False, fontsize=8,
              labelcolor=secondary_ink, loc="lower right", ncols=2)

    fig.tight_layout()
    out_path = OUT_DIR / "hardware_savings_ranking.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=surface)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
