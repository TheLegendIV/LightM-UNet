"""One-off recovery: reconstruct each config's result.csv from its
subprocess_output.txt training log, for sweeps run before
run_upscale_pareto.py wrote a per-config result.csv (see its main()
docstring on the shared-summary.csv race across Slurm array tasks this
was fixed to avoid -- a sweep run before that fix only has the LAST
config's row surviving in the old shared summary.csv, but every config's
raw subprocess_output.txt is still on disk, so nothing is actually lost).

Usage:
    python upscale/recover_results.py --results-dir upscale/results/pareto_e15
"""
from __future__ import annotations

import argparse
from pathlib import Path

from archive.upscale.pareto_common import ParsedMetrics, latest_training_log, parse_metrics_text, parse_training_log, repo_root
from archive.upscale.run_upscale_pareto import EXPERIMENTS, attach_param_budgets, write_csv


def recover_one(exp_dir: Path, exp: dict) -> dict | None:
    log_file = latest_training_log(exp_dir)
    if log_file is not None:
        metrics = parse_training_log(log_file)
    else:
        subprocess_output = exp_dir / "subprocess_output.txt"
        if not subprocess_output.exists():
            return None
        metrics = parse_metrics_text(subprocess_output.read_text(errors="replace"))
    if not metrics.pseudo_dice:
        return None

    mean_epoch_time = sum(metrics.epoch_times) / len(metrics.epoch_times) if metrics.epoch_times else None
    return {
        "id": exp["id"], "name": exp["name"], "track": exp["track"],
        "channels": ",".join(str(c) for c in exp["channels"]),
        "bottlenecks": ",".join(str(c) for c in exp["bottlenecks"]),
        "decoder_type": exp["decoder_type"],
        "params": exp["params"], "params_m": exp["params_m"],
        "hypothesis": exp["hypothesis"],
        "returncode": 0, "duration_s": "",
        "mean_epoch_time_s": mean_epoch_time,
        "final_dice": metrics.final_dice, "best_dice": metrics.best_dice,
        "momentum": metrics.momentum, "acceleration": metrics.acceleration,
        "num_dice_epochs": len(metrics.pseudo_dice),
        "pseudo_dice_by_epoch": ";".join(f"{v:.6f}" for v in metrics.pseudo_dice),
        "results_root": str(exp_dir),
    }


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Recover per-config result.csv from subprocess_output.txt logs.")
    parser.add_argument("--results-dir", type=Path, default=root / "upscale" / "results" / "pareto_e15")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    by_id = {e["id"]: e for e in attach_param_budgets(EXPERIMENTS)}

    recovered, skipped = [], []
    for exp_dir in sorted(args.results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        config_id = exp_dir.name.split("_", 1)[0]
        exp = by_id.get(config_id)
        if exp is None:
            continue  # not one of this table's config folders
        row = recover_one(exp_dir, exp)
        if row is None:
            skipped.append(config_id)
            continue
        write_csv([row], exp_dir / "result.csv")
        recovered.append(config_id)

    print(f"Recovered {len(recovered)} result.csv files: {recovered}")
    if skipped:
        print(f"Skipped (no parseable log -- still running, failed, or empty): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
