"""FIFO BRAM18K of the four constrained 150 FPS builds (MILP/artifacts/S12_512_ablation_v2) BEFORE the FIFO simulation (the ILP's analytical FIFO model) and AFTER it (simulation-sized FIFOs) -> s12_512_fifo_memory.png
+ fifo_memory_before_after.csv.

  before  <solve>/<arm>/layer_bits_folding_*.json: `intra_block_fifos` + `inter_block_fifos` (the analytically sized FIFOs: the 85 intra-block FIFOs deeper than the minimum, the inter-block FIFOs at depth 2).
  after   <solve>/<arm>/simfifo/layer_bits_folding_*_simfifo.json: the same two lists after net_explicit.py (every FIFO of the dataflow, ~500, sized by the block / block-pair / whole-net simulation).
  Plot = the BRAM18 count of those FIFOs (the rest of the FIFOs sit in LUTs / SRLs); the CSV also has the allocated memory (depth x width, kbit) and the FIFO LUTs. The INT8 arm is the relaxed_fps150 solve (the
  40% cap one is infeasible).

Usage:  python analysis/report/MILP/constrained_builds/plot_fifo_memory.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "compression" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, MUTED, _style_axes  # noqa: E402
from plot_uniform_vs_ilp_overlay import UNIFORM_COLOR  # noqa: E402

ARTIFACTS = REPO_ROOT / "MILP" / "artifacts" / "S12_512_ablation_v2"
BUILDS = [  # label, solve folder, ILP json, simulated json
    ("INT4", "cap40_fps150/uniform_int4", "layer_bits_folding_uniform_int4.json", "layer_bits_folding_int4_cap40_simfifo.json"),
    ("INT6", "cap40_fps150/uniform_int6", "layer_bits_folding_uniform_int6.json", "layer_bits_folding_int6_cap40_simfifo.json"),
    ("INT8", "relaxed_fps150/uniform_int8", "layer_bits_folding_uniform_int8.json", "layer_bits_folding_int8_fps150_simfifo.json"),
    ("Mixed 4/6/8", "cap40_fps150/mixed_468", "layer_bits_folding_mixed_468.json", "layer_bits_folding_mixed468_cap40_simfifo.json"),
]


def totals(d: dict) -> dict:
    intra, inter = d["intra_block_fifos"], d["inter_block_fifos"]
    bits = sum(f.get("depth_alloc", f["depth"]) * f["width_bits"] for f in intra + inter)
    return {"n_fifos": len(intra) + len(inter), "kbit": bits / 1000.0,
            "bram18": sum(f.get("mem_bram18", 0) for f in intra) + sum(f.get("bram18", 0) for f in inter),
            "lut": sum(f.get("mem_lut", 0) for f in intra) + sum(f.get("lut", 0) for f in inter)}


def load() -> list[dict]:
    out = []
    for label, solve, ilp, sim in BUILDS:
        a = totals(json.loads((ARTIFACTS / solve / ilp).read_text()))
        b = totals(json.loads((ARTIFACTS / solve / "simfifo" / sim).read_text()))
        out.append({"label": label, "before": a, "after": b})
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["build", "n_fifos_before", "n_fifos_after", "kbit_before", "kbit_after", "kbit_change_pct", "bram18_before", "bram18_after", "bram18_change", "fifo_lut_before", "fifo_lut_after"])
        for r in rows:
            a, b = r["before"], r["after"]
            w.writerow([r["label"], a["n_fifos"], b["n_fifos"], f"{a['kbit']:.1f}", f"{b['kbit']:.1f}", f"{100 * (b['kbit'] - a['kbit']) / a['kbit']:+.1f}", a["bram18"], b["bram18"],
                        f"{b['bram18'] - a['bram18']:+d}", f"{a['lut']:.0f}", f"{b['lut']:.0f}"])


def plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [("bram18", "FIFO BRAM18K (count)", "{:.0f}")]
    fig, axes = plt.subplots(1, 1, figsize=(7.5, 5.4), facecolor=SURFACE)
    axes = [axes]
    x = np.arange(len(rows))
    w = 0.36
    for ax, (key, ylabel, fmt) in zip(axes, panels):
        _style_axes(ax)
        before = [r["before"][key] for r in rows]
        after = [r["after"][key] for r in rows]
        b1 = ax.bar(x - w / 2, before, w, color=MUTED, edgecolor=SURFACE, zorder=3, label="Before sim")
        b2 = ax.bar(x + w / 2, after, w, color=UNIFORM_COLOR, edgecolor=SURFACE, zorder=3, label="After sim")
        for bars in (b1, b2):
            for rect in bars:
                ax.annotate(fmt.format(rect.get_height()), (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 3), textcoords="offset points", ha="center",
                            color=INK, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rows], fontweight="bold")
        ax.set_ylabel(ylabel, color=INK, fontsize=11, fontweight="bold")
        ax.set_title("FIFO BRAM18K Before vs. After Simulation", color=INK, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(before + after) * 1.12)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
        leg = ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7", fontsize=9, labelcolor=SECONDARY_INK)
        for t in leg.get_texts():
            t.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    rows = load()
    write_csv(rows, HERE / "fifo_memory_before_after.csv")
    print(open(HERE / "fifo_memory_before_after.csv").read())
    plot(rows, HERE / "s12_512_fifo_memory.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
