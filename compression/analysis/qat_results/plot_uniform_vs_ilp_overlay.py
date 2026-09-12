"""Overlays the "naive uniform quantization, same folding as alpha=0.25"
baseline (compression/hawq/uniform_bits_same_folding.py's own INT4/6/8
outputs, QAT'd via qat_12_dense_relu_warmstart150ep_uniform_int468_
samefolding_alpha0.25_array.job) onto the real per-layer joint-ILP alpha
sweep's own dice-vs-latency Pareto plot (plot_forcedsp_lut70_alpha_sweep.py)
-- answers "does the ILP's smarter per-layer bit allocation beat naive
uniform quantization at the SAME hardware folding structure?"

All three uniform points share the EXACT SAME latency (59.02ms) by
construction (folding copied unchanged from alpha=0.25 -- see that script's
own docstring for why cycles don't depend on bit-width at fixed folding),
so they stack vertically at one x-position. LUT% (which DOES vary sharply
with bit-width at fixed folding: INT4=54.4%, INT6=79.2%, INT8=114.0% -- the
real finding from that script's own run) is shown as a text annotation next
to each point, and INT8 (which exceeds 100% -- does not physically fit the
chip) gets a distinct marker (open square, dashed red edge) rather than a
plain filled circle, so "doesn't fit" reads at a glance rather than only in
the annotation text.

Usage:
    python compression/analysis/qat_results/plot_uniform_vs_ilp_overlay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_forcedsp_lut70_alpha_sweep import (  # noqa: E402
    INK, SECONDARY_INK, MUTED, SURFACE, ALPHA_COLORS, REPO_ROOT,
    _style_axes, load_final_dice, load_ilp_latency,
)

DEFAULT_CSV = REPO_ROOT / "compression" / "results.csv"
OUT_PATH = Path(__file__).resolve().parent / "out" / "12_dense_relu_warmstart150ep_uniform_vs_ilp_overlay.png"

CONFIG_PREFIX = "12_dense_relu_warmstart150ep"
ILP_DIR_PREFIX = "12_dense_relu_warmstart150ep"

# Real numbers from compression/hawq/uniform_bits_same_folding.py's own run
# (folding fixed to alpha=0.25's real solve) -- see that script's printed
# output; not re-derived here since it requires the real cost-model call.
UNIFORM_LUT_PCT = {4: 54.4, 6: 79.2, 8: 114.0}
UNIFORM_LATENCY_MS = 59.02  # identical for all three by construction (folding unchanged)
UNIFORM_COLOR = "#6a4c93"


def load_uniform_dice(csv_path: Path) -> dict[int, tuple[float, int]]:
    """Returns {bits: (dice, best_epoch)} by reading results.csv's own base
    (checkpoint_best) rows for the uniform_int{bits}_samefolding_alpha0.25
    QAT runs directly (same pattern as load_final_dice, but keyed by bits
    not alpha, and this trainer-name convention doesn't have a clean regex
    -- the three names are known in advance)."""
    import csv

    names = {
        4: "nnUNetTrainerLayerQuantENet_12_dense_relu_warmstart150ep_perlayer_12_dense_relu_warmstart150ep_uniform_int4_samefolding_alpha0.25_ft15ep",
        6: "nnUNetTrainerLayerQuantENet_12_dense_relu_warmstart150ep_perlayer_12_dense_relu_warmstart150ep_uniform_int6_samefolding_alpha0.25_ft15ep",
        8: "nnUNetTrainerLayerQuantENet_12_dense_relu_warmstart150ep_perlayer_12_dense_relu_warmstart150ep_uniform_int8_samefolding_alpha0.25_ft15ep",
    }
    dice_by_bits = {}
    with open(csv_path, newline="") as f:
        rows = {row["config_name"]: row for row in csv.DictReader(f)}
    for bits, name in names.items():
        if name in rows:
            dice_by_bits[bits] = float(rows[name]["dice"]), int(float(rows[name]["epochs"]))
    return dice_by_bits


def main() -> int:
    import matplotlib.pyplot as plt

    ilp_dice = load_final_dice(DEFAULT_CSV, CONFIG_PREFIX, "dice")
    ilp_summary = REPO_ROOT / "compression" / "hawq" / "artifacts" / f"{ILP_DIR_PREFIX}_ILP_outputs_perlayer_forcedsp_lut70" / "summary.csv"
    ilp_latency = load_ilp_latency(ilp_summary)
    uniform_dice = load_uniform_dice(DEFAULT_CSV)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=SURFACE)
    _style_axes(ax)

    # -- Real per-layer joint-ILP alpha sweep (the "smart" curve). --
    alphas = sorted(set(ilp_dice) & set(ilp_latency))
    xs = [ilp_latency[a] for a in alphas]
    ys = [ilp_dice[a] for a in alphas]
    ax.plot(xs, ys, color=MUTED, linewidth=1.3, linestyle="--", zorder=2)
    # alpha=0.25 sits almost exactly under uniform INT8 (0.7836 vs 0.7835) --
    # push its own label DOWN specifically, everything else alternates up/down.
    manual_y_off = {0.25: -30}
    for i, (a, x, y) in enumerate(zip(alphas, xs, ys)):
        color = ALPHA_COLORS.get(a, MUTED)
        ax.scatter([x], [y], color=color, s=100, zorder=4, edgecolor=SURFACE, linewidth=1.2,
                   label=f"ILP alpha={a}")
        y_off = manual_y_off.get(a, 10 if i % 2 == 0 else -16)
        ax.annotate(f"alpha={a}\n(LUT 70.0%)", (x, y), xytext=(6, y_off), textcoords="offset points",
                    color=color, fontsize=8, fontweight="bold")

    # -- Naive uniform baseline, folding pinned to alpha=0.25. --
    for bits in (4, 6, 8):
        if bits not in uniform_dice:
            continue
        dice, best_epoch = uniform_dice[bits]
        lut_pct = UNIFORM_LUT_PCT[bits]
        fits = lut_pct <= 100.0
        marker_kwargs = dict(s=130, zorder=5, linewidth=2)
        if fits:
            ax.scatter([UNIFORM_LATENCY_MS], [dice], color=UNIFORM_COLOR, marker="D",
                       edgecolor=SURFACE, **marker_kwargs)
        else:
            ax.scatter([UNIFORM_LATENCY_MS], [dice], facecolor="none", edgecolor="#c0392b",
                       marker="s", linestyle="--", **marker_kwargs)
        epoch_note = f", best@ep{best_epoch}" if best_epoch < 15 else ""
        fit_note = f"{lut_pct:.1f}% LUT" if fits else f"{lut_pct:.1f}% LUT -- DOES NOT FIT"
        # INT8 sits almost exactly under alpha=0.25's own marker -- push its
        # label well clear (up and left, with a leader line) instead of the
        # standard tight offset every other uniform point uses.
        if bits == 8:
            ax.annotate(f"uniform INT{bits}\n{fit_note}{epoch_note}", (UNIFORM_LATENCY_MS, dice),
                        xytext=(-70, 55), textcoords="offset points", color="#c0392b",
                        fontsize=8, fontweight="bold", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#c0392b", linewidth=0.8, alpha=0.6))
        else:
            ax.annotate(f"uniform INT{bits}\n{fit_note}{epoch_note}", (UNIFORM_LATENCY_MS, dice),
                        xytext=(-95, 0), textcoords="offset points", color=UNIFORM_COLOR if fits else "#c0392b",
                        fontsize=8, fontweight="bold", ha="left")

    ax.axvline(UNIFORM_LATENCY_MS, color=UNIFORM_COLOR, linewidth=0.8, linestyle=":", alpha=0.4, zorder=1)

    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.012)  # headroom for the INT8 leader-line annotation above the top cluster

    ax.set_xlabel("ILP-predicted latency (ms @ 100MHz)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("dice (checkpoint_best.pth)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        "12_dense_relu_warmstart150ep: per-layer ILP bit allocation vs. naive uniform quantization\n"
        "uniform INT4/6/8 folding PINNED to alpha=0.25's real solve -- same latency, LUT% varies sharply",
        color=INK, fontsize=10.5,
    )
    handles = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=UNIFORM_COLOR, markeredgecolor=SURFACE,
                   markersize=10, label="uniform, fits chip (≤100% LUT)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor="#c0392b",
                   markersize=10, label="uniform, DOES NOT FIT (>100% LUT)"),
    ]
    legend1 = ax.legend(handles=handles, loc="lower right", frameon=True, facecolor=SURFACE,
                         edgecolor="#c3c2b7", fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.add_artist(legend1)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")

    print("\nSummary:")
    print(f"{'point':>16} {'dice':>8} {'latency_ms':>12} {'LUT%':>8}")
    for a in alphas:
        print(f"{'alpha='+str(a):>16} {ilp_dice[a]:>8.4f} {ilp_latency[a]:>12.2f} {'70.0':>8}")
    for bits in (4, 6, 8):
        if bits in uniform_dice:
            dice, best_epoch = uniform_dice[bits]
            print(f"{'uniform INT'+str(bits):>16} {dice:>8.4f} {UNIFORM_LATENCY_MS:>12.2f} {UNIFORM_LUT_PCT[bits]:>8.1f}"
                  + (f"  (best@epoch{best_epoch})" if best_epoch < 15 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
