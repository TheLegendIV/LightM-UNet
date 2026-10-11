"""DSR ablation, 256x256 S12 dense nearest-upsample (noconv) ReLU net, uniform INT6, partition 2 only,
min-resources MILP, target 250 FPS, mvau_wwidth_max 72, simulated FIFO depths: real post-synth
LUT/LUTRAM/DSP/BRAM (% of XCZU7EV) per DSR setting vs the FINN auto-fold control. FPS above each group.
Same bar style as plot_dsr_ablation_hw.py, without the FIFO split.

Source (only): hardware/results.csv, model S12_dense_256_u4_analytical_v1, configs ratchet_<pct>_simfifo_milpfold_...
and ratchet_ablation_finn_autofold_... ("ratchet" = old name of DSR). FPS is its estimated_throughput_fps column.
Checked: DSP (25/20/15/15/15, FINN 20) equals the MILP's partition-2 DSP estimate for every arm.

Usage: python analysis/report/MILP/plot_dsr_ablation_256_hw.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "compression" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, REPO_ROOT, _style_axes  # noqa: E402

RESULTS = REPO_ROOT / "hardware/results.csv"
OUT = Path(__file__).resolve().parent / "figures" / "dsr_ablation_256_hw_resources.png"
OUT_CSV = Path(__file__).resolve().parent / "dsr_ablation_256_hw_table.csv"
MODEL = "S12_dense_256_u4_analytical_v1"

LUT_TOT, DSP_TOT, BRAM36_TOT = 230400, 1728, 312  # XCZU7EV
# (label, results.csv config)
VARIANTS = [
    ("DSR\n1%", "ratchet_1pct_simfifo_milpfold_partition_2_ooc_synth"),
    ("25%", "ratchet_25pct_simfifo_milpfold_partition_2_ooc_synth"),
    ("100%", "ratchet_100pct_simfifo_milpfold_partition_2_ooc_synth"),
    ("200%", "ratchet_200pct_simfifo_milpfold_partition_2_ooc_synth"),
    ("DSR\noff", "ratchet_off_simfifo_milpfold_partition_2_ooc_synth"),
    ("FINN\nautofold", "ratchet_ablation_finn_autofold_autofold_partition_2_ooc_synth"),
]
SERIES = [("LUT", "#0061d6"), ("LUTRAM", "#7fb0ee"), ("DSP", "#eb4300"), ("BRAM", "#00a78c")]


def load_rows() -> list[dict]:
    """Raw post-synth counts (BRAM in 36K tiles, the CSV's BRAM column) plus their % of XCZU7EV."""
    with open(RESULTS, newline="") as f:
        by_cfg = {r["config"]: r for r in csv.DictReader(f) if r["model_name"] == MODEL}
    rows = []
    for _, cfg in VARIANTS:
        r = by_cfg[cfg]
        lut, lutram, dsp, bram = (float(r[k]) for k in ("LUT", "LUTRAM", "DSP", "BRAM"))
        rows.append({"LUT_n": lut, "LUTRAM_n": lutram, "DSP_n": dsp, "BRAM_n": bram,
                     "LUT": 100 * lut / LUT_TOT, "LUTRAM": 100 * lutram / LUT_TOT,
                     "DSP": 100 * dsp / DSP_TOT, "BRAM": 100 * bram / BRAM36_TOT,
                     "fps": float(r["estimated_throughput_fps"])})
    return rows


def draw_bars(ax, rows, w) -> None:
    """Side by side: LUT | LUTRAM | DSP | BRAM, FPS above each group. LUTRAM is the subset of LUTs used as
    distributed memory (same % of XCZU7EV LUTs)."""
    n = range(len(rows))
    for j, (name, color) in enumerate(SERIES):
        ax.bar([i + (j - 1.5) * w for i in n], [r[name] for r in rows], w, color=color, label=name,
               edgecolor=SURFACE, zorder=3)
    top = [max(r[k] for k, _ in SERIES) for r in rows]
    for i, r in enumerate(rows):
        ax.text(i, top[i] + 0.4, f"{r['fps']:.0f} FPS", ha="center", va="bottom",
                color=INK, fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(top) * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=True, facecolor=SURFACE,
              edgecolor="#c3c2b7", fontsize=8.5, labelcolor=SECONDARY_INK)


def main() -> int:
    import matplotlib.pyplot as plt

    rows = load_rows()
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "LUT", "LUTRAM", "DSP", "BRAM_36K", "LUT_pct", "LUTRAM_pct", "DSP_pct", "BRAM_pct", "FPS"])
        for (lab, _), r in zip(VARIANTS, rows):
            wr.writerow([lab.replace("\n", " "), f"{r['LUT_n']:.0f}", f"{r['LUTRAM_n']:.0f}", f"{r['DSP_n']:.0f}",
                         f"{r['BRAM_n']:.0f}", f"{r['LUT']:.2f}", f"{r['LUTRAM']:.2f}", f"{r['DSP']:.2f}",
                         f"{r['BRAM']:.2f}", f"{r['fps']:.1f}"])
    print(f"Wrote {OUT_CSV}")

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    _style_axes(ax)
    draw_bars(ax, rows, 0.19)
    ax.set_xticks(range(len(rows)), [v[0] for v in VARIANTS])
    ax.set_ylabel("% of XCZU7EV", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("DSR Setting", color=INK, fontsize=10, fontweight="bold")
    ax.set_title("Post Synthesis Resource Utilization vs DSR at Fixed Target FPS and Uniform INT6 Quantization",
                 color=INK, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
