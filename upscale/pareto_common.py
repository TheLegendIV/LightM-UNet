"""Generic pareto-sweep harness -- cherry-picked from wip_pc's
lightm-unet/pareto_experiments/run_lmunet_pareto.py (the shared module its
other run_*_pareto.py scripts imported from).

This is the *methodology* the user asked to bring over: a "dirty" 15-epoch
sweep over an experiment table, one nnUNetv2_train subprocess per config,
scraping the training log's "Pseudo dice [...]" lines for a per-epoch dice
curve, then a CSV + a params-vs-final-dice pareto plot (colored by momentum
= second-half-vs-first-half dice trend, a cheap proxy for "still improving
vs. plateaued/overfitting" used to help pick graduation candidates). Only
the harness is ported -- the actual architectures come from this repo's own
parametric `enet/nnunetv2/nets/ENet.py` (see run_upscale_pareto.py), not
wip_pc's bespoke ENetUpscaled/ENetUpscaleArch nets.

Not specific to any one trainer/config schema -- run_upscale_pareto.py
supplies the EXPERIMENTS table and the ENET_* env vars per run.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PSEUDO_DICE_RE = re.compile(r"Pseudo dice\s+\[([^\]]*)\]")
EPOCH_TIME_RE = re.compile(r"Epoch time:\s+([0-9.]+)\s+s")


@dataclass
class ParsedMetrics:
    pseudo_dice: list[float]
    final_dice: float | None
    best_dice: float | None
    momentum: float | None
    acceleration: float | None
    epoch_times: list[float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_float_list(raw: str) -> list[float]:
    # ast.literal_eval cannot parse 'nan'; replace with None sentinel first
    cleaned = re.sub(r"\bnan\b", "None", raw)
    values = ast.literal_eval("[" + cleaned + "]")
    return [float("nan") if v is None else float(v) for v in values]


def parse_metrics_text(text: str) -> ParsedMetrics:
    epoch_means: list[float] = []
    epoch_times: list[float] = []
    for line in text.splitlines():
        dice_match = PSEUDO_DICE_RE.search(line)
        if dice_match:
            dice_values = parse_float_list(dice_match.group(1))
            valid = [v for v in dice_values if v == v]  # nan != nan
            if valid:
                epoch_means.append(sum(valid) / len(valid))
        time_match = EPOCH_TIME_RE.search(line)
        if time_match:
            epoch_times.append(float(time_match.group(1)))

    if not epoch_means:
        return ParsedMetrics([], None, None, None, None, epoch_times)

    final_dice = epoch_means[-1]
    best_dice = max(epoch_means)
    if len(epoch_means) < 2:
        momentum = None
        acceleration = None
    else:
        mid = len(epoch_means) // 2
        momentum = final_dice - epoch_means[mid]
        early_den = max(mid, 1)
        late_den = max(len(epoch_means) - 1 - mid, 1)
        early_slope = (epoch_means[mid] - epoch_means[0]) / early_den
        late_slope = (epoch_means[-1] - epoch_means[mid]) / late_den
        acceleration = late_slope - early_slope

    return ParsedMetrics(epoch_means, final_dice, best_dice, momentum, acceleration, epoch_times)


def parse_training_log(log_file: Path) -> ParsedMetrics:
    return parse_metrics_text(log_file.read_text(errors="replace"))


def latest_training_log(results_root: Path) -> Path | None:
    logs = list(results_root.rglob("training_log*.txt"))
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def ensure_preprocessed(dataset_id: str, dataset_name: str, nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path) -> None:
    """Plan+preprocess `dataset_id` if it hasn't been already, and sync
    splits_final.json from raw -- the exact guard compression/slurm/*.job
    already has as shell, needed here too since run_upscale_pareto.py /
    graduate.py can be invoked directly (not just via the Slurm wrapper),
    e.g. on a fresh HPC checkout where Dataset501_ARCADE was never
    preprocessed: nnUNetv2_train fails deep inside dataloader setup
    (FileNotFoundError on nnUNetPlans_2d/train_*.pkl) rather than with a
    clear "not preprocessed" message, so it's worth guarding against
    up front instead of debugging that traceback every time."""
    plans_path = nnunet_preprocessed / dataset_name / "nnUNetPlans.json"
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    if not plans_path.exists():
        print(f"=== {plans_path} missing -- running nnUNetv2_plan_and_preprocess -d {dataset_id} ===")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity"],
            env=env, check=True,
        )
    else:
        print(f"=== Plans already exist at {plans_path}, skipping preprocessing ===")

    raw_splits = nnunet_raw / dataset_name / "splits_final.json"
    if raw_splits.exists():
        shutil.copy(raw_splits, nnunet_preprocessed / dataset_name / "splits_final.json")


def run_subprocess_and_parse(command: list[str], cwd: Path, env: dict, log_root: Path) -> tuple[ParsedMetrics, subprocess.CompletedProcess, float]:
    """Runs `command`, saves raw stdout, and returns (parsed_metrics, completed, duration_s)."""
    start = time.perf_counter()
    completed = subprocess.run(
        command, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    duration_s = time.perf_counter() - start
    print(completed.stdout)
    (log_root / "subprocess_output.txt").write_text(completed.stdout, errors="replace")

    log_file = latest_training_log(log_root)
    metrics = parse_training_log(log_file) if log_file is not None else ParsedMetrics([], None, None, None, None, [])
    if not metrics.pseudo_dice:
        metrics = parse_metrics_text(completed.stdout)
    return metrics, completed, duration_s
