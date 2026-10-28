"""Error (%) of the cost model against real out-of-context synthesis, per build, for LUT / BRAM18K / DSP, calibrated and uncalibrated (stock FINN) estimates.

  s12_512_cost_model_error.png   absolute error, 100 * |model - real| / real. render(True, ...) draws the signed version instead: 100 * (model - real) / real, + overestimate, - underestimate, zero line drawn.
x =average W/A bit width ((mean weight bits + mean activation bits) / 2 over the quantizer sites: INT4/6/8 -> 4/6/8, Mixed -> 7.41 from its per-site assignment); a faint vertical line per build, named on
the top axis. Six series: shape + colour = resource (blue circle LUT, orange square BRAM18K, aqua triangle DSP, same for both estimates), fill = estimate (hollow calibrated, as in the estimated plot; filled uncalibrated; the line
is also dashed for calibrated, solid for uncalibrated); the lines join all four builds in order of average width (INT4, INT6, Mixed, INT8). Mixed is a different kind of design (per-site mix, its own folding), so the
joins are a visual guide, not an interpolation. y is capped at 100 %.
Source: modelled_vs_real_cost.csv (plot_constrained_builds.py); signed values also in model_error_by_design.csv (build_error_summary.py).
Colours: slots 1-3 of the reference categorical palette (blue, orange, aqua), validated all-pairs; aqua is below 3:1 on the light surface, so identity also rides on the marker shape and the legend.
    python report/MILP/constrained_builds/plot_model_error.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "nas" / "analysis" / "qat_results"))
from plot_forcedsp_lut70_alpha_sweep import INK, SECONDARY_INK, SURFACE, _style_axes  # noqa: E402

sys.path.insert(0, str(HERE))
from plot_style import ALPHA, CAL_STYLE, LINE_ALPHA, LINE_WIDTH, MARK_SIZE, MARK_WIDTH, RESOURCE_STYLE, UNCAL_STYLE  # noqa: E402  (style shared with plot_constrained_builds.py's estimated plot)

# resource -> (marker, legend label, calibrated column, uncalibrated column, real column)
_COLUMNS = {"lut": ("model_lut", "model_uncal_lut", "real_lut"), "bram": ("model_bram18", "model_uncal_bram18", "real_bram18eq"), "dsp": ("model_dsp", "model_uncal_dsp", "real_dsp")}
RESOURCES = {r: (RESOURCE_STYLE[r][0], RESOURCE_STYLE[r][1], *cols) for r, cols in _COLUMNS.items()}
COLORS = {r: s[2] for r, s in RESOURCE_STYLE.items()}
BUILD_ORDER = ["INT4", "INT6", "INT8", "Mixed"]      # matched on the first word of the CSV's scheme name ("Mixed" or "Mixed 4/6/8")
build = lambda row: row["scheme"].split()[0]          # noqa: E731
MIXED_SITES = REPO_ROOT / "ILP" / "artifacts" / "S12_512_ablation_v2" / "cap40_fps150" / "mixed_468" / "layer_bits_SITES_mixed_468.json"


def avg_wa_bits(name: str) -> float:
    """x position = (mean weight bits + mean activation bits) / 2 over the quantizer sites. INTn is uniform W=A=n, so n; Mixed from its deployed per-site assignment (unweighted mean over sites)."""
    if name.startswith("INT"):
        return float(name[3:])
    sites = json.loads(MIXED_SITES.read_text())
    w, a = list(sites["layer_weight_bits"].values()), list(sites["layer_act_bits"].values())
    return (sum(w) / len(w) + sum(a) / len(a)) / 2


def render(signed: bool, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = sorted(csv.DictReader(open(HERE / "modelled_vs_real_cost.csv", newline="")), key=lambda r: BUILD_ORDER.index(build(r)))
    err = (lambda model, real: 100.0 * (float(model) - float(real)) / float(real)) if signed else (lambda model, real: 100.0 * abs(float(model) - float(real)) / float(real))

    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    _style_axes(ax)
    xs = [avg_wa_bits(build(r)) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: xs[i])          # all four builds joined in order of average W/A bit width: INT4, INT6, Mixed, INT8
    for z, (r, (marker, _, cal, unc, real)) in enumerate(RESOURCES.items()):
        color = COLORS[r]
        for col, style, face in ((unc, UNCAL_STYLE, color), (cal, CAL_STYLE, SURFACE)):        # uncalibrated: filled shape; calibrated: hollow (surface-filled, so the line does not show through), as in the estimated plot
            ys = [err(rows[i][col], rows[i][real]) for i in order]
            ax.plot([xs[i] for i in order], ys, linestyle=style, linewidth=LINE_WIDTH, color=color, alpha=LINE_ALPHA, zorder=3 + z)                       # connecting line
            ax.plot([xs[i] for i in order], ys, linestyle="none", marker=marker, markersize=MARK_SIZE, markerfacecolor=face, markeredgecolor=color, markeredgewidth=MARK_WIDTH, alpha=ALPHA, zorder=5 + z)   # markers on top
    for x in xs:        # a faint vertical line per build; the build names are on the top axis
        ax.axvline(x, color="#c3c2b7", linewidth=0.8, alpha=0.7, zorder=1)
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks(xs)
    top_axis.set_xticklabels([build(r) for r in rows], fontsize=9.5, fontweight="bold", color=SECONDARY_INK)
    top_axis.tick_params(length=0, pad=4)
    top_axis.spines["top"].set_visible(False)

    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_xlim(3.5, 8.5)
    lo = -100 if signed else 0
    ax.set_ylim(lo, 100)                                          # capped at 100 % (largest error: +92 %, -77 %)
    ax.set_yticks(range(lo, 101, 10 if not signed else 20))       # labelled every 10 % (every 20 % signed), a faint line every 5 %
    ax.set_yticks([v for v in range(lo, 100, 5) if v % (10 if not signed else 20)], minor=True)
    ax.grid(axis="y", which="minor", color="#e1e0d9", linewidth=0.5, alpha=0.8, zorder=0)
    ax.tick_params(axis="y", which="minor", length=0)
    if signed:
        ax.axhline(0, color=INK, linewidth=1.0, alpha=0.6, zorder=2)
    ax.set_ylabel("Error (%)" if signed else "Absolute Error (%)", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel("Average W/A Bit Width", color=INK, fontsize=11, fontweight="bold")
    ax.set_title("Cost Model Error vs. Synthesis", color=INK, fontsize=12, fontweight="bold", pad=26)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=COLORS[r], markeredgecolor=COLORS[r], markeredgewidth=MARK_WIDTH, markersize=MARK_SIZE) for r, (m, *_) in RESOURCES.items()]
    labels = [v[1] for v in RESOURCES.values()]
    for name, style, face in (("Calibrated", CAL_STYLE, SURFACE), ("Uncalibrated", UNCAL_STYLE, SECONDARY_INK)):
        handles.append(Line2D([], [], color=SECONDARY_INK, linestyle=style, linewidth=1.2, alpha=0.7, marker="o", markersize=MARK_SIZE, markerfacecolor=face, markeredgecolor=SECONDARY_INK, markeredgewidth=MARK_WIDTH))
        labels.append(name)
    leg = ax.legend(handles, labels, loc="upper right", ncol=2, columnspacing=1.2, frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7", fontsize=7.5, markerscale=0.75, borderpad=0.5, labelspacing=0.35,
                    labelcolor=SECONDARY_INK)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    render(False, HERE / "s12_512_cost_model_error.png")       # absolute error; render(True, ...) gives the signed version (+ over, - under)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
