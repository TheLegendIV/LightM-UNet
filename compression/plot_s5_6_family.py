"""Dice vs Params / Dice vs mem_elements / Dice vs MACs scatter plots for the
Stage 26 S5.6-derived probe family (compression/slurm/
stage_26_s5_6_probe_family_array.job), one figure per cost metric, with
S5-SeparableDense (5_6_separable_dense_dilation -- the config all 8 variants
derive from, see that job's own header comment) overlaid as a highlighted
reference point so every variant reads against the config it was actually
cost-screened from (compression/finn_cost_s5_6_variants.py's own INT8 FINN-R
estimates). Same scatter/log-x/annotation style as compression/
plot_stage_results.py, generalized to a 3rd cost metric (mem_elements).

"MACs" here is flops/2 (results.csv stores FLOPs = 2*MACs, the usual
multiply-accumulate convention -- see compression/utils.py's count_flops
docstring), plotted rather than raw FLOPs since MACs is the unit
finn_cost_s5_6_variants.py's own cost estimates were expressed in.

Usage:
    python compression/plot_s5_6_family.py
    python compression/plot_s5_6_family.py --results-csv compression/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE = "26_s5_6_probe_family"
BASELINE_CONFIG = "nnUNetTrainerENet_5_6_separable_dense_dilation"

# Same muted palette as compression/plot_stage_results.py, for visual
# consistency across this directory's plots.
INK, SECONDARY_INK, MUTED, GRID, BASELINE_COLOR, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)
POINT_COLOR = "#2a78d6"
BASELINE_MARKER_COLOR = "#eb6834"

METRICS = {
    "params": dict(column="params", xlabel="Parameters", filename="dice_vs_params.png"),
    "mem_elements": dict(column="mem_elements", xlabel="FINN buffer elements", filename="dice_vs_mem_elements.png"),
    "macs": dict(column=None, xlabel="MACs", filename="dice_vs_macs.png"),
}


def display_name(config_name: str) -> str:
    if config_name == BASELINE_CONFIG:
        return "S5.6 (baseline)"
    return config_name.replace("nnUNetTrainerENet_26_", "V").replace("nnUNetTrainerENet_", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dice vs Params/mem_elements/MACs scatter plots for the Stage 26 S5.6 probe family.")
    parser.add_argument("--results-csv", type=Path, default=REPO_ROOT / "compression" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "compression" / "results" / "s5_6_family")
    return parser.parse_args()


def x_values(spec: dict, df: pd.DataFrame) -> pd.Series:
    if spec["column"] is not None:
        return df[spec["column"]]
    return df["flops"] / 2  # MACs = FLOPs / 2, see module docstring


def plot_metric(name: str, spec: dict, df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    xs = x_values(spec, df)
    valid = xs.notna() & df["dice"].notna()
    if not valid.any():
        print(f"[{name}] no rows with a parsed value -- skipping.")
        return
    plot_df = df[valid].copy()
    plot_df["_x"] = xs[valid]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BASELINE_COLOR)
    ax.tick_params(colors=MUTED, labelsize=9)

    baseline_row = plot_df[plot_df["config_name"] == BASELINE_CONFIG]
    variant_rows = plot_df[plot_df["config_name"] != BASELINE_CONFIG]

    if not baseline_row.empty:
        ax.scatter(baseline_row["_x"], baseline_row["dice"], marker="D", color=BASELINE_MARKER_COLOR,
                    s=110, zorder=4, edgecolors="black", linewidths=0.6, label="S5.6 (baseline)")

    ax.scatter(variant_rows["_x"], variant_rows["dice"], color=POINT_COLOR, s=80, zorder=3,
               edgecolors="black", linewidths=0.5)

    for _, row in plot_df.iterrows():
        ax.annotate(display_name(row["config_name"]), (row["_x"], row["dice"]), fontsize=7,
                    color=SECONDARY_INK, textcoords="offset points", xytext=(6, 5))

    ax.set_xlabel(spec["xlabel"], color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Dice", color=SECONDARY_INK, fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc="best")
    ax.set_title(f"S5.6 probe family (stage {STAGE}): Dice vs {spec['xlabel']}", color=INK, fontsize=12)
    fig.tight_layout()
    out_path = out_dir / spec["filename"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    args = parse_args()
    if not args.results_csv.exists():
        print(f"{args.results_csv} not found.")
        return 1

    df = pd.read_csv(args.results_csv)
    stage_df = df[df["stage"] == STAGE]
    baseline_df = df[df["config_name"] == BASELINE_CONFIG]
    if stage_df.empty:
        print(f"No rows found for stage={STAGE!r} in {args.results_csv}.")
        return 1
    combined = pd.concat([baseline_df, stage_df], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in METRICS.items():
        plot_metric(name, spec, combined, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
