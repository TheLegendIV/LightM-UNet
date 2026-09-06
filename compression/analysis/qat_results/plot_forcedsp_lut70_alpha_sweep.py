"""Plots compression/results.csv's own real 15-epoch QAT results for the
per-LAYER joint bits+folding ILP sweep at a HARD 70% LUT cap under the
forced-DSP derating (compression/slurm/qat_<prefix>_joint_alpha_sweep_15ep_
perlayer_candidatebits468_forcedsp_lut70_array.job -- 5 alpha points x 3
checkpoint epochs, one config family at a time, e.g. "12_separable_dense_
relu" (S12) or "27_2_reg_trailing" (S27.2)).

Two figures:
  1. <prefix>_forcedsp_lut70_dice_trend.png -- dice vs. checkpoint epoch
     milestone (5/10/15), one line per alpha, same house style as
     plot_bnfreeze_dice_trend.py's own dice-vs-epoch convention (single BN
     variant here -- this sweep always used ENET_FREEZE_BN=0).
  2. <prefix>_forcedsp_lut70_dice_vs_latency.png -- the actual accuracy/
     latency Pareto tradeoff this alpha sweep exists to explore: epoch-15
     dice vs. each alpha's own ILP-predicted latency_ms_at_100mhz (from
     compression/hawq/artifacts/<Prefix>_ILP_outputs_perlayer_forcedsp_lut70/
     summary.csv), all 5 points sharing the SAME real hard 70% LUT budget.

Only the explicit *_epoch{5,10,15} rows are used for the trend plot -- NOT
the base (no-suffix) row, which reflects whichever epoch had the best EMA
pseudo-dice (checkpoint_best.pth), not a fixed epoch number (see
plot_bnfreeze_dice_trend.py's own note on this).

Usage:
    python compression/analysis/qat_results/plot_forcedsp_lut70_alpha_sweep.py \\
        --config-prefix 12_separable_dense_relu --ilp-dir-prefix S12
    python compression/analysis/qat_results/plot_forcedsp_lut70_alpha_sweep.py \\
        --config-prefix 27_2_reg_trailing --ilp-dir-prefix S27_2
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "compression" / "results.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"
EPOCHS = (5, 10, 15)

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
ALPHA_COLORS = {
    0.0: "#2a78d6", 0.25: "#3fa796", 0.5: "#eb6834", 0.75: "#c44536", 1.0: "#8a5fbf",
}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def load_epoch_trend(csv_path: Path, config_prefix: str, metric: str) -> dict[float, dict[int, float]]:
    pattern = re.compile(
        rf"_{re.escape(config_prefix)}_joint_alpha(?P<alpha>[0-9.]+)_perlayer_candidatebits468_"
        rf"forcedsp_lut70_ft15ep_epoch(?P<epoch>5|10|15)$"
    )
    data: dict[float, dict[int, float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            m = pattern.search(row["config_name"])
            if not m:
                continue
            alpha = float(m.group("alpha"))
            epoch = int(m.group("epoch"))
            data.setdefault(alpha, {})[epoch] = float(row[metric])
    return data


def load_final_dice(csv_path: Path, config_prefix: str, metric: str) -> dict[float, float]:
    """The base (no-epoch-suffix) row -- checkpoint_best.pth's own EMA-best
    dice, used only for the Pareto plot's y-axis (a single "how good did
    this alpha's run end up" point, not a trend), never mixed with the fixed
    -epoch trend data above."""
    pattern = re.compile(
        rf"_{re.escape(config_prefix)}_joint_alpha(?P<alpha>[0-9.]+)_perlayer_candidatebits468_"
        rf"forcedsp_lut70_ft15ep$"
    )
    data: dict[float, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            m = pattern.search(row["config_name"])
            if not m:
                continue
            data[float(m.group("alpha"))] = float(row[metric])
    return data


def load_ilp_latency(ilp_summary_csv: Path) -> dict[float, float]:
    data: dict[float, float] = {}
    with open(ilp_summary_csv, newline="") as f:
        for row in csv.DictReader(f):
            data[float(row["alpha"])] = float(row["latency_ms_at_100mhz"])
    return data


def plot_dice_trend(data: dict[float, dict[int, float]], metric: str, title_prefix: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    alphas = sorted(data.keys())
    missing = [(a, e) for a in alphas for e in EPOCHS if e not in data.get(a, {})]
    if missing:
        print(f"Note: missing (alpha, epoch) rows (skipped): {missing}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=SURFACE)
    _style_axes(ax)

    for alpha in alphas:
        color = ALPHA_COLORS.get(alpha, MUTED)
        series = data.get(alpha, {})
        xs = [e for e in EPOCHS if e in series]
        ys = [series[e] for e in xs]
        if xs:
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                     markeredgecolor=SURFACE, markeredgewidth=1, zorder=3, label=f"alpha={alpha}")

    ax.set_xticks(EPOCHS)
    ax.set_xlim(min(EPOCHS) - 0.5, max(EPOCHS) + 0.5)
    ax.set_xlabel("checkpoint epoch", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel(metric, color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        f"{title_prefix}: {metric} vs. QAT epoch, per-layer joint ILP alpha sweep\n"
        "hard 70% LUT cap, forced-DSP derating, candidate bits {4,6,8} -- fixed ENET_SEED/calibration-seed across alphas",
        color=INK, fontsize=10.5,
    )
    ax.legend(loc="center right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_dice_vs_latency(dice: dict[float, float], latency: dict[float, float], title_prefix: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    alphas = sorted(set(dice) & set(latency))
    missing = sorted(set(dice) ^ set(latency))
    if missing:
        print(f"Note: alpha(s) missing from one of dice/latency (skipped): {missing}")

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    _style_axes(ax)

    xs = [latency[a] for a in alphas]
    ys = [dice[a] for a in alphas]
    ax.plot(xs, ys, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    for i, (a, x, y) in enumerate(zip(alphas, xs, ys)):
        ax.scatter([x], [y], color=ALPHA_COLORS.get(a, MUTED), s=90, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        y_off = 10 if i % 2 == 0 else -14
        ax.annotate(f"alpha={a}", (x, y), xytext=(6, y_off), textcoords="offset points",
                    color=ALPHA_COLORS.get(a, MUTED), fontsize=9, fontweight="bold")

    ax.set_xlabel("ILP-predicted latency (ms @ 100MHz)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("best-checkpoint dice (checkpoint_best.pth)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        f"{title_prefix}: accuracy vs. latency tradeoff at a fixed 70% LUT budget\n"
        "(forced-DSP derating, candidate bits {4,6,8} -- every point fits the SAME real chip budget)",
        color=INK, fontsize=10.5,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_summary(data: dict[float, dict[int, float]], dice: dict[float, float], latency: dict[float, float], metric: str) -> None:
    alphas = sorted(data.keys())
    print(f"\n{metric} vs. epoch, per alpha:")
    print(f"{'alpha':>6} " + " ".join(f"{f'ep{e}':>10}" for e in EPOCHS))
    for a in alphas:
        vals = [data[a].get(e) for e in EPOCHS]
        print(f"{a:>6} " + " ".join(f"{v:>10.4f}" if v is not None else f"{'--':>10}" for v in vals))

    print(f"\nbest-checkpoint dice vs. ILP-predicted latency:")
    print(f"{'alpha':>6} {'dice':>8} {'latency_ms':>12}")
    for a in sorted(set(dice) | set(latency)):
        d = dice.get(a)
        l = latency.get(a)
        print(f"{a:>6} {d if d is None else f'{d:.4f}':>8} {l if l is None else f'{l:.2f}':>12}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--config-prefix", required=True,
                         help="e.g. '12_separable_dense_relu' (S12) or '27_2_reg_trailing' (S27.2) -- must match "
                              "the run_name convention used by the forced-DSP-70%%-LUT alpha-sweep job.")
    parser.add_argument("--ilp-dir-prefix", required=True,
                         help="e.g. 'S12' or 'S27_2' -- selects compression/hawq/artifacts/<prefix>_ILP_outputs_"
                              "perlayer_forcedsp_lut70/summary.csv for the latency Pareto plot.")
    parser.add_argument("--metric", default="dice", help="Which results.csv column to plot (default: dice).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    trend_data = load_epoch_trend(args.csv, args.config_prefix, args.metric)
    if not trend_data:
        print(f"No matching epoch-trend rows found in {args.csv} for config-prefix {args.config_prefix!r}.")
        return 1

    final_dice = load_final_dice(args.csv, args.config_prefix, args.metric)
    ilp_summary = REPO_ROOT / "compression" / "hawq" / "artifacts" / f"{args.ilp_dir_prefix}_ILP_outputs_perlayer_forcedsp_lut70" / "summary.csv"
    latency = load_ilp_latency(ilp_summary) if ilp_summary.exists() else {}
    if not latency:
        print(f"Note: no ILP summary found at {ilp_summary} -- skipping the dice-vs-latency plot.")

    print_summary(trend_data, final_dice, latency, args.metric)

    plot_dice_trend(trend_data, args.metric, args.ilp_dir_prefix, args.out_dir / f"{args.config_prefix}_forcedsp_lut70_{args.metric}_trend.png")
    if latency:
        plot_dice_vs_latency(final_dice, latency, args.ilp_dir_prefix, args.out_dir / f"{args.config_prefix}_forcedsp_lut70_{args.metric}_vs_latency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
