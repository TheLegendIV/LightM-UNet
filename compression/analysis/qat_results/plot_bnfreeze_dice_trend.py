"""Plots compression/results.csv's own real S12 15-epoch QAT results
(nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock_12_
separable_dense_relu_joint_alpha<A>_maxlat1000ms_ft15ep_{bnfreeze,
nobnfreeze}_epoch{5,10,15} -- 5 alpha points x 2 BN variants x 3 checkpoint
epochs, 30 rows) as dice-vs-epoch-milestone, BN-freeze vs no-freeze
overlaid, to answer two questions at a glance: (1) does freezing BatchNorm
during this short QAT fine-tune help, hurt, or wash out, and (2) at which
epoch milestone does that ranking already look stable, so future sweeps
don't need to run the full 15 epochs to know which BN setting to pick.

Only the explicit *_epoch{5,10,15} rows are used -- NOT the base (no-suffix)
row, which reflects whichever epoch had the best EMA pseudo-dice
(checkpoint_best.pth), not a fixed epoch number, so it isn't a comparable
x-axis point (see this session's own "epochs=1 anomaly" diagnosis: that
row's real epoch is data-dependent, per run).

Usage:
    python compression/analysis/qat_results/plot_bnfreeze_dice_trend.py
    python compression/analysis/qat_results/plot_bnfreeze_dice_trend.py --metric dice_binary
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "compression" / "results.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

ROW_PATTERN = re.compile(
    r"_joint_alpha(?P<alpha>[0-9.]+)_maxlat1000ms_ft15ep_(?P<variant>bnfreeze|nobnfreeze)_epoch(?P<epoch>5|10|15)$"
)
EPOCHS = (5, 10, 15)

INK, SECONDARY_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb",
)
# Fixed categorical order, one hue per alpha -- reuses plot_block_bits.py's
# own FREE_COLOR (blue)/FIXED_COLOR (orange) at alpha=0.0/0.5 for visual
# continuity with that companion script's palette.
ALPHA_COLORS = {
    0.0: "#2a78d6", 0.25: "#3fa796", 0.5: "#eb6834", 0.75: "#c44536", 1.0: "#8a5fbf",
}
BNFREEZE_MEAN_COLOR = "#0b0b0b"
NOBNFREEZE_MEAN_COLOR = "#0b6e4f"


def load_results(csv_path: Path, metric: str) -> dict[str, dict[float, dict[int, float]]]:
    """Returns {"bnfreeze": {alpha: {epoch: metric_value}}, "nobnfreeze": {...}}."""
    data: dict[str, dict[float, dict[int, float]]] = {"bnfreeze": {}, "nobnfreeze": {}}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            m = ROW_PATTERN.search(row["config_name"])
            if not m:
                continue
            alpha = float(m.group("alpha"))
            variant = m.group("variant")
            epoch = int(m.group("epoch"))
            data[variant].setdefault(alpha, {})[epoch] = float(row[metric])
    return data


def make_figure(data: dict[str, dict[float, dict[int, float]]], metric: str):
    import matplotlib.pyplot as plt

    alphas = sorted(data["bnfreeze"].keys())
    missing = [
        (variant, alpha, e) for variant in ("bnfreeze", "nobnfreeze") for alpha in alphas for e in EPOCHS
        if e not in data[variant].get(alpha, {})
    ]
    if missing:
        print(f"Note: missing rows (skipped from lines that touch them): {missing}")

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)

    # Thin, per-alpha context lines -- solid=bnfreeze, dashed=nobnfreeze.
    for alpha in alphas:
        color = ALPHA_COLORS.get(alpha, MUTED)
        for variant, linestyle, alpha_line in (("bnfreeze", "-", 0.55), ("nobnfreeze", "--", 0.55)):
            series = data[variant].get(alpha, {})
            xs = [e for e in EPOCHS if e in series]
            ys = [series[e] for e in xs]
            if xs:
                ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.3, alpha=alpha_line,
                         marker="o", markersize=4, zorder=2)

    # Bold mean-across-alphas lines -- the actual answer to "which is better, and since when".
    for variant, color, linestyle, label in (
        ("bnfreeze", BNFREEZE_MEAN_COLOR, "-", "bnfreeze (mean of 5 alphas)"),
        ("nobnfreeze", NOBNFREEZE_MEAN_COLOR, "--", "no bnfreeze (mean of 5 alphas)"),
    ):
        means = []
        for e in EPOCHS:
            vals = [data[variant][a][e] for a in alphas if e in data[variant].get(a, {})]
            means.append(sum(vals) / len(vals) if vals else None)
        xs = [e for e, v in zip(EPOCHS, means) if v is not None]
        ys = [v for v in means if v is not None]
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=3, marker="o", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=1, zorder=5, label=label)
        if xs:
            ax.annotate(label.split(" (")[0], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                         color=color, fontsize=9, fontweight="bold", va="center")

    ax.set_xticks(EPOCHS)
    ax.set_xlim(min(EPOCHS) - 1, max(EPOCHS) + 2.5)
    ax.set_xlabel("checkpoint epoch", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel(metric, color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        "BN freeze vs. no-freeze during 15-epoch QAT fine-tuning (S12, per-block joint bits+folding)\n"
        "thin lines = one alpha each; bold = mean across all 5 alphas",
        color=INK, fontsize=11,
    )

    alpha_handles = [
        plt.Line2D([0], [0], color=ALPHA_COLORS.get(a, MUTED), linewidth=2, label=f"alpha={a}")
        for a in alphas
    ]
    variant_handles = [
        plt.Line2D([0], [0], color=MUTED, linestyle="-", linewidth=2, label="bnfreeze"),
        plt.Line2D([0], [0], color=MUTED, linestyle="--", linewidth=2, label="no bnfreeze"),
    ]
    legend1 = ax.legend(handles=alpha_handles, loc="lower right", frameon=False, fontsize=8.5,
                         labelcolor=SECONDARY_INK, title="alpha (color)", title_fontsize=8.5)
    ax.add_artist(legend1)
    ax.legend(handles=variant_handles, loc="upper left", frameon=False, fontsize=8.5,
              labelcolor=SECONDARY_INK, title="BN variant (line style)", title_fontsize=8.5)

    fig.tight_layout()
    return fig


def print_summary(data: dict[str, dict[float, dict[int, float]]], metric: str) -> None:
    alphas = sorted(data["bnfreeze"].keys())
    print(f"\n{metric} -- mean across {len(alphas)} alphas, bnfreeze vs no-bnfreeze, per epoch milestone:")
    print(f"{'epoch':>6} {'bnfreeze':>10} {'no_bnfreeze':>12} {'diff (bnfreeze-no)':>20}  winner")
    for e in EPOCHS:
        bf_vals = [data["bnfreeze"][a][e] for a in alphas if e in data["bnfreeze"].get(a, {})]
        nf_vals = [data["nobnfreeze"][a][e] for a in alphas if e in data["nobnfreeze"].get(a, {})]
        if not bf_vals or not nf_vals:
            continue
        bf_mean, nf_mean = sum(bf_vals) / len(bf_vals), sum(nf_vals) / len(nf_vals)
        diff = bf_mean - nf_mean
        winner = "bnfreeze" if diff > 0 else "no_bnfreeze" if diff < 0 else "tie"
        print(f"{e:>6} {bf_mean:>10.4f} {nf_mean:>12.4f} {diff:>+20.4f}  {winner}")

    print(f"\nPer-alpha detail ({metric}):")
    print(f"{'alpha':>6} {'epoch':>6} {'bnfreeze':>10} {'no_bnfreeze':>12} {'diff':>10}  winner")
    for a in alphas:
        for e in EPOCHS:
            bf = data["bnfreeze"].get(a, {}).get(e)
            nf = data["nobnfreeze"].get(a, {}).get(e)
            if bf is None or nf is None:
                continue
            diff = bf - nf
            winner = "bnfreeze" if diff > 0 else "no_bnfreeze" if diff < 0 else "tie"
            print(f"{a:>6} {e:>6} {bf:>10.4f} {nf:>12.4f} {diff:>+10.4f}  {winner}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--metric", default="dice", help="Which results.csv column to plot (default: dice).")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output PNG path (default: compression/analysis/qat_results/out/bnfreeze_<metric>_trend.png).")
    args = parser.parse_args()

    data = load_results(args.csv, args.metric)
    if not data["bnfreeze"] and not data["nobnfreeze"]:
        print(f"No matching rows found in {args.csv} for pattern {ROW_PATTERN.pattern!r}.")
        return 1

    print_summary(data, args.metric)

    fig = make_figure(data, args.metric)
    out_path = args.out or DEFAULT_OUT_DIR / f"bnfreeze_{args.metric}_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
