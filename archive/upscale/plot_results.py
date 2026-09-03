"""Aggregate upscale/ pareto sweep results into one CSV, print per-track
summary stats, and plot Dice vs. params colored by momentum (heat).

Reads every config's own `result.csv` (one per subfolder, written by
run_upscale_pareto.py's main() loop) rather than trusting a shared
summary.csv -- when the sweep runs as one Slurm array task per config (the
normal HPC path), concurrent tasks would otherwise race to overwrite a
single shared file. This script is the safe place to combine them, run any
time after some or all configs have finished (partial results are fine).

Usage:
    python upscale/plot_results.py
    python upscale/plot_results.py --results-dir upscale/results/pareto_e15
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from archive.upscale.pareto_common import repo_root
from archive.upscale.run_upscale_pareto import RESULT_FIELDNAMES


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for result_csv in sorted(results_dir.glob("*/result.csv")):
        with result_csv.open(newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDNAMES})


def _float_or_none(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def print_group_stats(rows: list[dict]) -> None:
    groups: dict[str, dict[str, list[float]]] = {}
    n_skipped = 0
    for row in rows:
        dice = _float_or_none(row.get("final_dice"))
        params_m = _float_or_none(row.get("params_m"))
        if dice is None or params_m is None:
            n_skipped += 1
            continue
        bucket = groups.setdefault(row["track"], {"dice": [], "params_m": []})
        bucket["dice"].append(dice)
        bucket["params_m"].append(params_m)

    print("\n=== Per-track summary (mean over configs with a parsed Dice) ===")
    header = f"{'track':<20}{'n':>4}{'mean_dice':>12}{'mean_params_m':>16}"
    print(header)
    print("-" * len(header))
    for track in sorted(groups):
        vals = groups[track]
        n = len(vals["dice"])
        mean_dice = sum(vals["dice"]) / n
        mean_params = sum(vals["params_m"]) / n
        print(f"{track:<20}{n:>4}{mean_dice:>12.4f}{mean_params:>16.4f}")
    if n_skipped:
        print(f"({n_skipped} configs skipped -- no parsed Dice yet, still running or failed.)")


def plot(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot: {exc}")
        return

    plotted = [r for r in rows if _float_or_none(r.get("final_dice")) is not None]
    if not plotted:
        print("No rows with a parsed final_dice -- nothing to plot.")
        return

    track_markers = {"max_unpool": "o", "upsample_conv": "s", "bottleneck_depth": "^"}
    params_m = [_float_or_none(r["params_m"]) for r in plotted]
    dice = [_float_or_none(r["final_dice"]) for r in plotted]
    momentum = [_float_or_none(r.get("momentum")) or 0.0 for r in plotted]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = None
    for track, marker in track_markers.items():
        idx = [i for i, r in enumerate(plotted) if r["track"] == track]
        if not idx:
            continue
        sc = ax.scatter(
            [params_m[i] for i in idx], [dice[i] for i in idx],
            c=[momentum[i] for i in idx], cmap="inferno",
            vmin=min(momentum), vmax=max(momentum) if max(momentum) > min(momentum) else min(momentum) + 1e-6,
            marker=marker, s=70, zorder=3, label=track, edgecolors="black", linewidths=0.4,
        )
        for i in idx:
            ax.annotate(plotted[i]["id"], (params_m[i], dice[i]), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Momentum (2nd-half vs 1st-half dice trend)")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Final mean pseudo Dice")
    ax.set_title("Upscale pareto sweep: Dice vs Params, colored by momentum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Aggregate upscale/ pareto results into one CSV + plot.")
    parser.add_argument("--results-dir", type=Path, default=root / "upscale" / "results" / "pareto_e15")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_results(args.results_dir)
    if not rows:
        print(f"No result.csv files found under {args.results_dir}/*/result.csv")
        return 1

    summary_path = args.results_dir / "summary.csv"
    write_summary(rows, summary_path)
    print(f"Aggregated {len(rows)} configs into {summary_path}")

    print_group_stats(rows)
    plot(rows, args.results_dir / "pareto_final_dice.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
