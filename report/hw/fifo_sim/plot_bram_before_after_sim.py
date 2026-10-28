"""S12 dense bilinear-512: total BRAM18K of each configuration BEFORE the FIFO simulation (the MILP's own estimate) and AFTER it (nodes + the simulated FIFO lists of net_explicit.py), as grouped
bars, with the simulated FPS above each 'after' bar (the whole-net chain simulation's last-frame period at 100 MHz); the MILP estimate's own FPS is not labelled.
Same house style as nas/analysis/qat_results/plot_alpha025_vs_uniform_int48.py.

Data: ILP/artifacts/S12_dense_512_uniform_cap1_v1/<arm>/layer_bits_folding_<arm>.json (_diagnostics.total_bram18k_calibrated, bottleneck_cycles) and
      <arm>_simfifo/layer_bits_folding_<arm>_simfifo.json (_diagnostics.total_bram18k_with_all_fifos, fifo_sim.chain.last.period). BRAM is shown as percent of the XCZU7EV's 624 BRAM18K.

Usage:
    python report/MILP/fifo_sim/plot_bram_before_after_sim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "nas" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, ALPHA_COLORS, _style_axes  # noqa: E402
from plot_uniform_vs_ilp_overlay import UNIFORM_COLOR  # noqa: E402

ARM_DIR = REPO_ROOT / "ILP" / "artifacts" / "S12_dense_512_uniform_cap1_v1"
ARMS = [("INT4", "uniform_int4"), ("INT6", "uniform_int6"), ("INT8", "uniform_int8"), ("Mixed 6/8", "mixed_68")]
CLOCK_HZ = 100e6
OUT_PATH = HERE / "bram_before_after_sim.png"


def load(arm: str) -> dict:
    milp = json.loads((ARM_DIR / arm / f"layer_bits_folding_{arm}.json").read_text())["_diagnostics"]
    sim = json.loads((ARM_DIR / f"{arm}_simfifo" / f"layer_bits_folding_{arm}_simfifo.json").read_text())
    budget = sim["_diagnostics"]["xczu7ev_bram18k_budget"]
    return dict(
        before=100 * milp["total_bram18k_calibrated"] / budget, after=100 * sim["_diagnostics"]["total_bram18k_with_all_fifos"] / budget,
        before_n=milp["total_bram18k_calibrated"], after_n=sim["_diagnostics"]["total_bram18k_with_all_fifos"],
        fps_before=CLOCK_HZ / milp["bottleneck_cycles"], fps_after=CLOCK_HZ / sim["fifo_sim"]["chain"]["last"]["period"],
    )


def main() -> int:
    import matplotlib.pyplot as plt

    data = {name: load(arm) for name, arm in ARMS}
    fig, ax = plt.subplots(figsize=(9.0, 5.8), facecolor=SURFACE)
    _style_axes(ax)
    w = 0.38
    xs = list(range(len(ARMS)))
    for i, (name, _arm) in enumerate(ARMS):
        d = data[name]
        for dx, key, fps, color in ((-w / 2, "before", d["fps_before"], UNIFORM_COLOR), (w / 2, "after", d["fps_after"], ALPHA_COLORS.get(0.25))):
            ax.bar(i + dx, d[key], width=w, color=color, zorder=3, edgecolor=SURFACE, label=("MILP estimate" if key == "before" else "After simulation") if i == 0 else None)
            if key == "after":                                         # only the simulated (chain) FPS is shown; the MILP's own FPS is not labelled
                ax.annotate(f"{fps:.1f}", (i + dx, d[key]), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _ in ARMS])
    ax.set_ylabel("BRAM18K Consumption (%)", color=INK, fontsize=11, fontweight="bold")
    ax.set_title("BRAM Before vs. After FIFO Simulation", color=INK, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(max(d["before"], d["after"]) for d in data.values()) * 1.2)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    leg = ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7", fontsize=9, labelcolor=SECONDARY_INK)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    for name, d in data.items():
        print(f"  {name:10s} BRAM18K {d['before_n']:.0f} -> {d['after_n']:.0f} ({d['before']:.1f}% -> {d['after']:.1f}%)   FPS {d['fps_before']:.2f} -> {d['fps_after']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
