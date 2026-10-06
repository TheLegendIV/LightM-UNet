"""DSR ablation (S12 dense nearest-upsample wm, partition 2 only, uniform INT4,
min-resources objective, target FPS 305.17, mvau_wwidth_max 80): real
post-synth LUT/BRAM/DSP (% of XCZU7EV) + FIFO-only BRAM per DSR setting, vs
FINN auto-fold control. FPS printed above each group (rtlsim stable throughput).

Sources:
  MILP/artifacts/S12_dense_dsr_ablation_v1/<tag>/summary.csv  (estimated FPS)
  hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs/
    finn_deployment_outputs/*_<tag>_*partition2*/report/{ooc_synth_and_timing,
    rtlsim_performance}.json, node_resource_calibration.csv (per-node FIFO real BRAM)
    real_dsp_overrides.json (real DSP; JSON DSP field is always 0, known parser bug)

Usage: python analysis/report/MILP/plot_dsr_ablation_hw.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "compression" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, REPO_ROOT, _style_axes  # noqa: E402

BUILD = REPO_ROOT / "hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs"
ART = REPO_ROOT / "MILP/artifacts/S12_dense_dsr_ablation_v1"
OUT = Path(__file__).resolve().parent / "figures" / "dsr_ablation_hw_resources.png"
OUT_CSV = Path(__file__).resolve().parent / "dsr_ablation_hw_table.csv"
# U4 analytical-flow partition 2 (non-deadlocked full 8-way build), real post-synth row of the ratchet-ablation dump.
ANALYTICAL_CSV = REPO_ROOT / "hardware/builds/S12_dense_256_ratchet_ablation_v1/resource_dump_partition2.csv"
ANALYTICAL_ARM = "partition2_baseline"

LUT_TOT, DSP_TOT, BRAM36_TOT = 230400, 1728, 312  # XCZU7EV
# (label, build tag, artifact dir or None)
VARIANTS = [
    ("Min. feasible\n(DSR 1.02)", "dsrmin_1x", "dsrmin_1x"),
    ("2x", "dsrmin_2x", "dsrmin_2x"),
    ("4x", "dsrmin_4x", "dsrmin_4x"),
    ("8x", "dsrmin_8x", "dsrmin_8x"),
    ("DSR\noff", "dsr_off", "dsr_off"),
    ("FINN\nautofold", "dsr_ablation_autofold_control", None),
]
SERIES = [("LUT", "#0061d6"), ("LUTRAM", "#7fb0ee"), ("BRAM", "#00a78c"), ("DSP", "#eb4300"), ("FIFO BRAM", "#5600bf")]


def build_dir(tag: str) -> Path:
    hits = [p for p in (BUILD / "finn_deployment_outputs").iterdir()
            if f"_wm_{tag}_" in p.name and "partition2" in p.name]
    return sorted(hits)[-1]


# Partition 2 of the 8-way split = stage2.0..2.4 (5 regular bottlenecks: 15 MVAU, 5 AddStreams, 5 Dup, 5 SWU).
# Verified: the build's real MVAU PE/SIMD match the MILP's stage2.0..2.4 per_layer folds with 0 mismatches
# (dsrmin_1x/2x/8x). The partition also holds down2's two trailing join thresholds (residual_add, out_act)
# and its out_act Dup, but NOT stage2.4's own residual_add/out_act/Dup (those open partition 3).
P2_PREFIXES = tuple(f"stage2.{i}." for i in range(5))
P2_EXTRA = ("down2.residual_add", "down2.out_act", "down2.out_act.dup")
P2_EXCLUDE = ("stage2.4.residual_add", "stage2.4.out_act", "stage2.4.out_act.dup")
BRAM18_TOT = 624


def partition2_estimate(art_dir: Path) -> dict:
    """MILP's own per-node calibrated LUT/BRAM18/DSP/cycles summed over partition 2's nodes only
    (per_layer + extra_nodes of the solve's layer_bits_folding_<name>.json)."""
    d = json.load(open(art_dir / f"layer_bits_folding_{art_dir.name}.json"))
    lut = bram18 = dsp = 0.0
    cycles = 0
    for key, lut_k, dsp_k in (("per_layer", "lut_calibrated", "total_dsp"), ("extra_nodes", "lut_calibrated", "dsp")):
        for name, v in d[key].items():
            if (name.startswith(P2_PREFIXES) and name not in P2_EXCLUDE) or name in P2_EXTRA:
                lut += v[lut_k]
                bram18 += v["bram18k_calibrated"]
                dsp += v[dsp_k]
                cycles = max(cycles, v["cycles"])
    return {"est_LUT": 100 * lut / LUT_TOT, "est_BRAM": 100 * bram18 / BRAM18_TOT,
            "est_DSP": 100 * dsp / DSP_TOT, "est_fps": 1e8 / cycles}


def load(tag: str, art: str | None) -> dict:
    d = build_dir(tag) / "report"
    ooc = json.load(open(d / "ooc_synth_and_timing.json"))
    rtl = json.load(open(d / "rtlsim_performance.json"))
    dsp = json.load(open(BUILD / "real_dsp_overrides.json"))[tag]
    fifo = fifo_bits = 0.0
    with open(d / "node_resource_calibration.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["node_kind"] == "FIFO":
                fifo_bits += float(r["depth"] or 0) * float(r["bits"] or 0)
                fifo += float(r["real_BRAM36"] or 0) + float(r["real_BRAM18"] or 0) / 2
    est = partition2_estimate(ART / art) if art else {}
    return {
        "LUT": 100 * ooc["LUT"] / LUT_TOT,
        "LUTRAM": 100 * ooc["LUTRAM"] / LUT_TOT,
        "BRAM": 100 * ooc["BRAM"] / BRAM36_TOT,
        "DSP": 100 * dsp / DSP_TOT,
        "FIFO BRAM": 100 * fifo / BRAM36_TOT,
        "fps": rtl["stable_throughput[images/s]"], "fifo_bits": fifo_bits, **est,
    }


def load_analytical() -> dict:
    """Real LUT/LUTRAM/DSP of the U4 analytical partition 2 from the dump CSV. Not in the dump (so the bar is
    skipped, value None): FIFO BRAM, FPS, and hence BRAM Computation (the dump's BRAM total includes the FIFOs)."""
    with open(ANALYTICAL_CSV, newline="") as f:
        r = next(r for r in csv.DictReader(f) if r["arm"] == ANALYTICAL_ARM)
    return {"LUT": 100 * float(r["LUT"]) / LUT_TOT, "LUTRAM": 100 * float(r["LUTRAM"]) / LUT_TOT,
            "DSP": 100 * float(r["DSP_real"]) / DSP_TOT, "BRAM": None, "FIFO BRAM": None,
            "fps": None, "fifo_bits": None}


TABLE_HEADER = [
    "est_LUT_pct", "real_LUT_pct", "LUT_err_pct",
    "est_DSP_pct", "real_DSP_pct", "DSP_err_pct",
    "est_BRAM_pct", "real_BRAM_excl_FIFO_pct", "BRAM_err_pct",
    "real_FIFO_BRAM_pct", "total_fifo_bits", "real_LUTRAM_pct",
    "est_FPS", "real_FPS", "FPS_err_pct",
]


def table_row(r) -> list:
    """Real BRAM here excludes FIFO BRAM (the MILP's BRAM estimate does not model FIFOs).
    err_pct = (real - est) / est * 100; blank when there is no estimate."""
    def trio(est, real):
        if real is None:
            return ["", "", ""]
        if est is None:
            return ["", f"{real:.2f}", ""]
        err = f"{100 * (real - est) / est:.1f}" if est else "n/a (est = 0)"
        return [f"{est:.2f}", f"{real:.2f}", err]
    g = r.get
    return (trio(g("est_LUT"), r["LUT"]) + trio(g("est_DSP"), r["DSP"])
            + trio(g("est_BRAM"), None if r["BRAM"] is None else r["BRAM"] - r["FIFO BRAM"])
            + ["" if r["FIFO BRAM"] is None else f"{r['FIFO BRAM']:.2f}",
               "" if r["fifo_bits"] is None else f"{r['fifo_bits']:.0f}", f"{r['LUTRAM']:.2f}"]
            + trio(g("est_fps"), r["fps"]))


def write_table(path, id_header, ids, rows) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(id_header + TABLE_HEADER)
        for i, r in zip(ids, rows):
            wr.writerow(i + table_row(r))


def draw_bars(ax, rows, w) -> None:
    """Side by side: LUT | LUTRAM | DSP | BRAM Computation (total - FIFO) | BRAM FIFO, FPS above each group.
    LUTRAM is the subset of LUTs used as distributed memory (same % of XCZU7EV LUTs)."""
    col = dict(SERIES)
    n = range(len(rows))
    bars = [
        ("LUT", col["LUT"], [r["LUT"] for r in rows]),
        ("LUTRAM", col["LUTRAM"], [r["LUTRAM"] for r in rows]),
        ("DSP", col["DSP"], [r["DSP"] for r in rows]),
        ("BRAM Computation", col["BRAM"], [None if r["BRAM"] is None else r["BRAM"] - r["FIFO BRAM"] for r in rows]),
        ("BRAM FIFO", col["FIFO BRAM"], [r["FIFO BRAM"] for r in rows]),
    ]
    for j, (name, color, vals) in enumerate(bars):  # a missing resource (None) skips that bar
        idx = [i for i in n if vals[i] is not None]
        ax.bar([i + (j - 2) * w for i in idx], [vals[i] for i in idx], w, color=color, label=name,
               edgecolor=SURFACE, zorder=3)
    top = [max(v[i] for _, _, v in bars if v[i] is not None) for i in n]
    for i, r in enumerate(rows):
        if r["fps"] is not None:
            ax.text(i, top[i] + 1.0, f"{r['fps']:.0f} FPS", ha="center", va="bottom",
                    color=INK, fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(top) * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=True, facecolor=SURFACE,
              edgecolor="#c3c2b7", fontsize=8.5, labelcolor=SECONDARY_INK)


def main() -> int:
    import matplotlib.pyplot as plt

    variants = VARIANTS + [("Analytical", None, None)]
    rows = [load(t, a) for _, t, a in VARIANTS] + [load_analytical()]
    for (lab, *_), r in zip(variants, rows):
        print(lab.replace("\n", " "), {k: round(v, 2) if v else v for k, v in r.items()})

    write_table(OUT_CSV, ["variant"], [[lab.replace("\n", " ")] for lab, *_ in variants], rows)
    print(f"Wrote {OUT_CSV}")

    w = 0.16
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    _style_axes(ax)
    draw_bars(ax, rows, w)
    ax.set_xticks(range(len(rows)), [v[0] for v in variants])
    ax.set_ylabel("% of XCZU7EV", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("DSR Setting", color=INK, fontsize=10, fontweight="bold")
    ax.set_title("Post Synthesis Resource Utilization vs DSR at Fixed Target FPS and Uniform INT4 Quantization", color=INK, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
