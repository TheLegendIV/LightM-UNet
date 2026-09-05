"""Summarizes and plots an alpha sweep of joint_bits_folding_ilp.py runs
(e.g. compression/hawq/artifacts/S12_ILP_outputs/'s own
block_bits_folding_12_separable_dense_relu_joint_alpha<A>_*.json files) --
one row per alpha in a summary.csv, plus three depth-profile plots (bit-width
and folding are both real "how does the choice vary across the network's
own depth" questions, best read as a profile, not a table):

  1. avg_bits_<...>.png  -- per BLOCK, x=network depth order (block_
     sensitivity's own key order, matching plot_block_bits.py's own
     DEFAULT_ORDER_FROM convention), y=(weight_bits+act_bits)/2, one line
     per alpha.
  2. pe_<...>.png / simd_<...>.png -- per LAYER (finer than block -- a
     folding decision is made per real Conv2d/ConvTranspose2d/MaxPool2d, not
     per block), x=network depth order (the FIRST input file's own per_layer
     key order -- dump_block_layer_geometry's real forward-execution order,
     preserved by dict insertion order, identical across every alpha in one
     sweep since they all trace the same architecture), y=PE or SIMD.

This is the alpha-SWEEP companion to plot_block_bits.py (which plots ONE
run's own weight/act bit profile) -- overlaying every alpha on the same
depth axis is what actually answers "does the profile SHAPE change with
alpha, or just shift up/down uniformly."

Usage:
    python compression/analysis/sensitivity/plot_alpha_sweep_depth.py \\
        --inputs compression/hawq/artifacts/S12_ILP_outputs/block_bits_folding_12_separable_dense_relu_joint_alpha*.json \\
        --out-dir compression/hawq/artifacts/S12_ILP_outputs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORDER_FROM = REPO_ROOT / "compression/hawq/artifacts/block_sensitivity_12_separable_dense_relu.json"
XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}

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


def _block_order(block_names: list[str], order_from: Path) -> list[str]:
    if order_from.exists():
        ref_order = list(json.loads(order_from.read_text()).keys())
        ordered = [n for n in ref_order if n in block_names]
        leftover = [n for n in block_names if n not in ordered]
        return ordered + leftover
    return block_names


def _group_boundaries(names: list[str]) -> list[int]:
    def prefix(n: str) -> str:
        return n.split(".")[0]
    return [i for i in range(1, len(names)) if prefix(names[i]) != prefix(names[i - 1])]


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        data = json.loads(p.read_text())
        if data.get("status") != "Optimal":
            print(f"Note: {p.name} status={data.get('status')!r} (not Optimal) -- skipped from plots/summary.")
            continue
        data["_path"] = p
        runs.append(data)
    return sorted(runs, key=lambda d: d["alpha"])


def write_summary(runs: list[dict], out_csv: Path) -> None:
    rows = []
    for run in runs:
        diag = run["_diagnostics"]
        w = list(run["stage_weight_bits"].values())
        a = list(run["stage_act_bits"].values())
        rows.append({
            "alpha": run["alpha"],
            "status": run["status"],
            "avg_weight_bits": sum(w) / len(w),
            "avg_act_bits": sum(a) / len(a),
            "avg_bits": (sum(w) / len(w) + sum(a) / len(a)) / 2,
            "lut_pct_of_budget": diag.get("lut_pct_of_budget"),
            "bram_pct_of_budget": diag.get("bram_pct_of_budget"),
            "total_cycles": diag.get("total_cycles"),
            "latency_ms_at_100mhz": diag.get("total_cycles", 0) / 100e6 * 1000 if diag.get("total_cycles") else None,
            "n_binary_vars": diag.get("n_binary_vars"),
            "candidate_bits": diag.get("candidate_bits"),
            "force_serial": diag.get("force_serial"),
            "source_file": run["_path"].name,
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    for row in rows:
        print(f"  alpha={row['alpha']}: avg_bits={row['avg_bits']:.3f} (w={row['avg_weight_bits']:.3f}, "
              f"a={row['avg_act_bits']:.3f}), LUT={row['lut_pct_of_budget']:.1f}%, "
              f"BRAM={row['bram_pct_of_budget']:.1f}%, latency={row['latency_ms_at_100mhz']:.1f}ms@100MHz")


def plot_avg_bits_per_block(runs: list[dict], order_from: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    block_names = _block_order(list(runs[0]["stage_weight_bits"].keys()), order_from)
    x = list(range(len(block_names)))
    boundaries = _group_boundaries(block_names)

    fig, ax = plt.subplots(figsize=(max(11, 0.34 * len(block_names)), 5.5), facecolor=SURFACE)
    _style_axes(ax)
    for b in boundaries:
        ax.axvline(b - 0.5, color=GRID, linewidth=0.9, zorder=1)

    for run in runs:
        alpha = run["alpha"]
        color = ALPHA_COLORS.get(alpha, MUTED)
        w, a = run["stage_weight_bits"], run["stage_act_bits"]
        avg = [(w[n] + a[n]) / 2 for n in block_names]
        ax.plot(x, avg, color=color, linewidth=1.6, marker="o", markersize=4, zorder=3, label=f"alpha={alpha}")

    ax.set_xticks(x)
    ax.set_xticklabels(block_names, rotation=90, fontsize=7.5, family="monospace", color=SECONDARY_INK)
    ax.set_ylabel("avg bits ((weight+act)/2)", color=SECONDARY_INK, fontsize=10)
    ax.set_title("Per-block average quantization vs. network depth, across the alpha sweep",
                 color=INK, fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_folding_per_layer(runs: list[dict], field: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # Real forward-execution depth order -- dump_block_layer_geometry's own
    # per_layer insertion order, identical across every alpha in the same
    # sweep (same architecture traced every time), so the FIRST run's own
    # key order is a valid reference for all of them.
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
        vals = [run["per_layer"][n][field] for n in layer_names]
        ax.plot(x, vals, color=color, linewidth=1.3, marker="o", markersize=3.5, alpha=0.85, zorder=3,
                 label=f"alpha={alpha}")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=90, fontsize=6, family="monospace", color=SECONDARY_INK)
    ax.set_yscale("log", base=2)
    ax.set_ylabel(f"{ylabel} (log2 scale)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(f"Per-layer {ylabel} folding vs. network depth, across the alpha sweep",
                 color=INK, fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True,
                         help="Glob pattern(s) or explicit paths to block_bits_folding_*.json alpha-sweep files.")
    parser.add_argument("--order-from", type=Path, default=DEFAULT_ORDER_FROM)
    parser.add_argument("--out-dir", type=Path, required=True,
                         help="Directory to write summary.csv into (plots go in <out-dir>/plots/).")
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.inputs for p in glob.glob(pattern)})
    if not paths:
        print(f"No files matched {args.inputs!r}.")
        return 1
    runs = load_runs(paths)
    if not runs:
        print("No Optimal runs to summarize/plot.")
        return 1

    write_summary(runs, args.out_dir / "summary.csv")
    plots_dir = args.out_dir / "plots"
    plot_avg_bits_per_block(runs, args.order_from, plots_dir / "avg_bits_per_block_depth.png")
    plot_folding_per_layer(runs, "pe", "PE", plots_dir / "pe_per_layer_depth.png")
    plot_folding_per_layer(runs, "simd", "SIMD", plots_dir / "simd_per_layer_depth.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
