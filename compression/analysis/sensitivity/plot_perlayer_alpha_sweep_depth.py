"""Per-LAYER counterpart to plot_alpha_sweep_depth.py -- for an alpha sweep
of joint_bits_folding_ilp_perlayer.py runs (e.g. compression/hawq/artifacts/
S12_ILP_outputs_perlayer/'s own layer_bits_folding_*.json files, one
(weight_bits, act_bits, pe, simd) per real conv/pool LAYER rather than per
block), plots:

  1. avg_bits_per_layer_depth.png -- x=network depth order (this sweep's own
     first file's per_layer key order = real forward-execution order), y=
     (weight_bits+act_bits)/2, one line per alpha. Genuinely per-LAYER here
     (not aggregated to block level) -- this is real data the block-level
     ILP could never produce, since it forces every layer in a block to the
     same bits by construction.
  2. pe_per_layer_depth.png / simd_per_layer_depth.png -- same x-axis,
     y=PE or SIMD.

Usage:
    python compression/analysis/sensitivity/plot_perlayer_alpha_sweep_depth.py \\
        --inputs compression/hawq/artifacts/S12_ILP_outputs_perlayer/layer_bits_folding_12_separable_dense_relu_joint_alpha*_candidatebits468_maxlat1000ms.json \\
        --out-dir compression/hawq/artifacts/S12_ILP_outputs_perlayer
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

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
    ax.tick_params(colors=MUTED, labelsize=8)


def _group_boundaries(names: list[str]) -> list[int]:
    def prefix(n: str) -> str:
        return n.split(".")[0]
    return [i for i in range(1, len(names)) if prefix(names[i]) != prefix(names[i - 1])]


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        data = json.loads(p.read_text())
        if data.get("status") != "Optimal":
            print(f"Note: {p.name} status={data.get('status')!r} (not Optimal) -- skipped.")
            continue
        data["_path"] = p
        runs.append(data)
    return sorted(runs, key=lambda d: d["alpha"])


def plot_field_per_layer(runs: list[dict], field: str, ylabel: str, out_path: Path, log_scale: bool = False) -> None:
    import matplotlib.pyplot as plt

    layer_names = list(runs[0]["per_layer"].keys())
    x = list(range(len(layer_names)))
    boundaries = _group_boundaries(layer_names)

    fig, ax = plt.subplots(figsize=(max(13, 0.22 * len(layer_names)), 5.5), facecolor=SURFACE)
    _style_axes(ax)
    for b in boundaries:
        ax.axvline(b - 0.5, color=GRID, linewidth=0.9, zorder=1)

    for run in runs:
        alpha = run["alpha"]
        color = ALPHA_COLORS.get(alpha, MUTED)
        if field == "avg_bits":
            vals = [(run["per_layer"][n]["weight_bits"] + run["per_layer"][n]["act_bits"]) / 2 for n in layer_names]
        else:
            vals = [run["per_layer"][n][field] for n in layer_names]
        ax.plot(x, vals, color=color, linewidth=1.3, marker="o", markersize=3.5, alpha=0.85, zorder=3,
                 label=f"alpha={alpha}")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=90, fontsize=6, family="monospace", color=SECONDARY_INK)
    if log_scale:
        ax.set_yscale("log", base=2)
        ylabel = f"{ylabel} (log2 scale)"
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10)
    ax.set_title(f"Per-layer {ylabel} vs. network depth, across the alpha sweep (per-layer joint ILP)",
                 color=INK, fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        print(f"No files matched {args.inputs!r}.")
        return 1
    runs = load_runs(paths)
    if not runs:
        print("No Optimal runs to plot.")
        return 1

    plots_dir = args.out_dir / "plots"
    plot_field_per_layer(runs, "avg_bits", "avg bits ((weight+act)/2)", plots_dir / "avg_bits_per_layer_depth.png")
    plot_field_per_layer(runs, "pe", "PE", plots_dir / "pe_per_layer_depth.png", log_scale=True)
    plot_field_per_layer(runs, "simd", "SIMD", plots_dir / "simd_per_layer_depth.png", log_scale=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
