"""S12 dense bilinear-512, constrained 150 FPS builds (MILP/artifacts/S12_512_ablation_v2): Dice of every quantization scheme against its resource cost, MODELLED and REAL.

  s12_512_constrained_lut.png    Dice vs LUT %
  s12_512_constrained_bram.png   Dice vs BRAM18K %
  s12_512_constrained_dsp.png    Dice vs DSP %
  s12_512_constrained_combined.png  Dice vs the equal-weight mean of the LUT %, BRAM18K % and DSP %
  s12_512_constrained_ff.png     Dice vs FF % (real synthesis only: the ILP does not model flip-flops; 460,800 FF in the XCZU7EV; no cap line for now)

Per scheme (INT4, INT6, INT8, mixed 4/6/8; the FINN auto-fold INT6 build is not shown), two markers joined by a line:
  modelled (circle)  the FINAL modelled design: the ILP folding after the FIFO simulation, i.e. `_diagnostics.total_lut_with_all_fifos / total_bram18k_with_all_fifos / total_dsp` of
                     <solve>/<arm>/simfifo/layer_bits_folding_*_simfifo.json (the sim-sized FIFOs are included, so this is what was built);
  real (square)      the out-of-context synthesis total of the 8 partitions (synth_utilization.csv, `total` rows; BRAM = BRAM18 + 2 x BRAM36).
  Percent of the XCZU7EV (230,400 LUT / 624 BRAM18 / 1,728 DSP). The 40% cap line is the resource cap the cap40_fps150 solves are held to (LUT / BRAM / DSP each).
  INT8 does not fit the 0.4 caps at 150 FPS (cap40_fps150/uniform_int8 is infeasible), so its modelled point is the relaxed_fps150/uniform_int8 solve (red); its real cost is the int8_caps100_fps150 build.
  Real DSP = the DSP48E2 count of synth_utilization.csv (real_dsp_per_partition.json; the DSP field of FINN's OOC reports itself is always 0).
Dice: dice_scores.csv (QAT evaluation, FP32 reference dotted). Also written: modelled_vs_real_cost.csv (every number plotted, plus the raw MILP estimate before the FIFO simulation, plus the model_uncal_*
columns: the same final design priced by STOCK FINN v0.10.1's own estimators (no constant of this repo's cost model; MVAU_rtl and the stream nodes are 0 LUT, every Thresholding stage is URAM because the
bridge leaves depth_trigger_uram at 0), from build_uncalibrated_cost.py -> uncalibrated_cost.json; blank when that file is missing; not plotted).

Usage:
    python analysis/report/MILP/constrained_builds/build_dice_csv.py      # dice_scores.csv
    python analysis/report/MILP/constrained_builds/build_uncalibrated_cost.py   # uncalibrated_cost.json (MILP container: needs torch)
    python analysis/report/MILP/constrained_builds/plot_constrained_builds.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "compression" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, ALPHA_COLORS, _style_axes  # noqa: E402
from plot_uniform_vs_ilp_overlay import UNIFORM_COLOR  # noqa: E402

ARTIFACTS = REPO_ROOT / "MILP" / "artifacts" / "S12_512_ablation_v2"
BUDGET = {"lut": 230400.0, "bram": 624.0, "dsp": 1728.0, "ff": 460800.0, "uram": 96.0}        # XCZU7EV: LUT, BRAM18, DSP, FF, URAM288
CAP_PCT = 40.0
INFEASIBLE_COLOR = "#c0392b"
MIXED_COLOR = ALPHA_COLORS.get(0.25)

# scheme -> (label, solve folder, simfifo json, synth build id or None)
SCHEMES = {
    "int4": ("INT4", "cap40_fps150/uniform_int4", "layer_bits_folding_int4_cap40_simfifo.json", "int4_cap40"),
    "int6": ("INT6", "cap40_fps150/uniform_int6", "layer_bits_folding_int6_cap40_simfifo.json", "int6_cap40"),
    "int8": ("INT8", "relaxed_fps150/uniform_int8", "layer_bits_folding_int8_fps150_simfifo.json", "int8_caps100_fps150"),
    "mixed468": ("Mixed 4/6/8", "cap40_fps150/mixed_468", "layer_bits_folding_mixed468_cap40_simfifo.json", "mixed468_cap40"),
}
RESOURCES = {"lut": "LUT Consumption (%)", "bram": "BRAM18K Consumption (%)", "dsp": "DSP Consumption (%)", "combined": "Mean of LUT, BRAM18K, DSP Consumption (%)", "ff": "FF Consumption (%)"}
TITLE_WORD = {"lut": "LUT", "bram": "BRAM18K", "dsp": "DSP", "combined": "Combined", "ff": "FF"}


def load_costs() -> dict:
    synth = {}
    for r in csv.DictReader(open(HERE / "synth_utilization.csv", newline="")):
        if r["partition"] == "total":
            synth[r["build"]] = r
    uncal_path = HERE / "uncalibrated_cost.json"
    uncal = json.loads(uncal_path.read_text()) if uncal_path.exists() else None
    out = {}
    for key, (label, solve, sim_json, build) in SCHEMES.items():
        d = json.loads((ARTIFACTS / solve / "simfifo" / sim_json).read_text())["_diagnostics"]
        milp = next(csv.DictReader(open(ARTIFACTS / solve / "summary.csv", newline="")))
        c = {"label": label, "solve": solve,
             "model_lut": d["total_lut_with_all_fifos"], "model_bram": d["total_bram18k_with_all_fifos"] + d.get("total_uram18", 0.0), "model_dsp": float(d["total_dsp"]),
             "milp_lut_pct": float(milp["lut_pct_of_budget"]), "milp_bram_pct": float(milp["bram_pct_of_budget"]), "milp_dsp_pct": float(milp["dsp_pct_of_budget"]),
             "real_lut": None, "real_bram": None, "real_dsp": None, "real_ff": None, "model_ff": None}      # the ILP does not model flip-flops: FF has real points only
        if build and build in synth:
            c["real_lut"] = float(synth[build]["LUT"])
            c["real_bram"] = float(synth[build]["BRAM18_equiv"])
            c["real_dsp"] = float(synth[build]["DSP"])
            c["real_ff"] = float(synth[build]["FF"])
        c["model_uncal_lut"] = c["model_uncal_bram"] = c["model_uncal_dsp"] = c["model_uncal_uram"] = None
        if uncal and key in uncal:        # build_uncalibrated_cost.py: the same final design priced by STOCK FINN's own estimators (no constant of this repo's cost model)
            c["model_uncal_lut"], c["model_uncal_bram"], c["model_uncal_dsp"], c["model_uncal_uram"] = (uncal[key]["uncalibrated"][k] for k in ("lut", "bram18", "dsp", "uram"))
        out[key] = c
    return out


def write_cost_csv(costs: dict, path: Path) -> None:
    cols = ["scheme", "milp_solve", "milp_lut_pct", "milp_bram_pct", "milp_dsp_pct", "model_lut", "model_lut_pct", "real_lut", "real_lut_pct", "model_bram18", "model_bram_pct", "real_bram18eq",
            "real_bram_pct", "model_dsp", "model_dsp_pct", "real_dsp", "real_dsp_pct", "real_ff", "real_ff_pct",
            "model_uncal_lut", "model_uncal_lut_pct", "model_uncal_bram18", "model_uncal_bram_pct", "model_uncal_dsp", "model_uncal_dsp_pct", "model_uncal_uram", "model_uncal_uram_pct"]
    pct = lambda v, k: "" if v is None else f"{100 * v / BUDGET[k]:.2f}"  # noqa: E731
    num = lambda v: "" if v is None else f"{v:.0f}"                         # noqa: E731
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, c in costs.items():
            w.writerow([c["label"], c["solve"], f"{c['milp_lut_pct']:.2f}", f"{c['milp_bram_pct']:.2f}", f"{c['milp_dsp_pct']:.2f}",
                        num(c["model_lut"]), pct(c["model_lut"], "lut"), num(c["real_lut"]), pct(c["real_lut"], "lut"),
                        num(c["model_bram"]), pct(c["model_bram"], "bram"), num(c["real_bram"]), pct(c["real_bram"], "bram"),
                        num(c["model_dsp"]), pct(c["model_dsp"], "dsp"), num(c["real_dsp"]), pct(c["real_dsp"], "dsp"), num(c["real_ff"]), pct(c["real_ff"], "ff"),
                        num(c["model_uncal_lut"]), pct(c["model_uncal_lut"], "lut"), num(c["model_uncal_bram"]), pct(c["model_uncal_bram"], "bram"),
                        num(c["model_uncal_dsp"]), pct(c["model_uncal_dsp"], "dsp"), num(c["model_uncal_uram"]), pct(c["model_uncal_uram"], "uram")])


def load_dice() -> tuple[dict, float]:
    rows = {r["scheme"]: r for r in csv.DictReader(open(HERE / "dice_scores.csv", newline=""))}
    return {k: float(rows[k]["dice"]) for k in SCHEMES}, float(rows["fp32"]["dice"])



def pct(c: dict, kind: str, resource: str):
    """Percent of the device for one resource, or the equal-weight mean of LUT / BRAM / DSP for 'combined' (None when a real value is missing)."""
    if resource == "combined":
        vals = [pct(c, kind, r) for r in ("lut", "bram", "dsp")]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)
    v = c[f"{kind}_{resource}"]
    return None if v is None else 100 * v / BUDGET[resource]


def plot_one(resource: str, costs: dict, dice: dict, fp32: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    _style_axes(ax)
    ax.axhline(fp32, color=INK, linestyle=":", linewidth=1.2, zorder=1, label=f"FP32 baseline ({fp32:.4f})")
    if resource != "ff":            # every capped resource has the same 40% cap, so it also bounds the mean; FF has no cap for now
        ax.axvline(CAP_PCT, color=INK, linewidth=1.2, alpha=0.6, zorder=1, label="40% resource cap")
    any_real = any_model = False
    for key, c in costs.items():
        model_pct, real_pct = pct(c, "model", resource), pct(c, "real", resource)
        color = MIXED_COLOR if key == "mixed468" else (INFEASIBLE_COLOR if key == "int8" else UNIFORM_COLOR)   # INT8 does not fit the 40% caps
        if model_pct is None and real_pct is None:      # nothing to plot (INT8 real cost is blank; FF has no model)
            continue
        if real_pct is not None:
            any_real = True
            if model_pct is not None:
                ax.plot([model_pct, real_pct], [dice[key]] * 2, color=color, linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)
            ax.scatter([real_pct], [dice[key]], color=color, s=120, marker="s", edgecolor=SURFACE, linewidth=1.4, zorder=5)
        if model_pct is not None:
            any_model = True
            ax.scatter([model_pct], [dice[key]], color=color, s=130, marker="o", edgecolor=SURFACE, linewidth=1.4, zorder=6)
        x_label = model_pct if model_pct is not None else real_pct
        ax.annotate(c["label"], (x_label, dice[key]), xytext=(8, -20) if (key == "mixed468" and resource == "combined") else (8, 9), textcoords="offset points", ha="left", color=color, fontsize=10, fontweight="bold")
    ax.set_xlabel(RESOURCES[resource], color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel("Dice", color=INK, fontsize=11, fontweight="bold")
    ax.set_title(f"Dice vs. {TITLE_WORD[resource]} Consumption", color=INK, fontsize=12, fontweight="bold")
    lo, hi = ax.get_xlim()
    ax.set_xlim(max(0, lo), max(hi, CAP_PCT + 3))
    ax.set_ylim(top=0.8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    handles, labels = ax.get_legend_handles_labels()
    if any_model:
        handles.append(Line2D([], [], marker="o", color="none", markerfacecolor=SECONDARY_INK, markersize=9))
        labels.append("Modelled (ILP + FIFO sim)")
    if any_real:
        handles.append(Line2D([], [], marker="s", color="none", markerfacecolor=SECONDARY_INK, markersize=9))
        labels.append("Real (OOC synthesis)")
    leg = ax.legend(handles, labels, loc="best", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7", fontsize=9, labelcolor=SECONDARY_INK)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    costs = load_costs()
    write_cost_csv(costs, HERE / "modelled_vs_real_cost.csv")
    print(open(HERE / "modelled_vs_real_cost.csv").read())
    dice, fp32 = load_dice()
    for resource in RESOURCES:
        plot_one(resource, costs, dice, fp32, HERE / f"s12_512_constrained_{resource}.png")
    for key, c in costs.items():
        print(f"combined modelled {c['label']}: {pct(c, 'model', 'combined'):.2f}%  real: {pct(c, 'real', 'combined')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
