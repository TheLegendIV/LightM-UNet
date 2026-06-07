from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


EXPERIMENTS = [
    {
        "id": "B0",
        "name": "baseline",
        "channels": (12, 20, 32, 44, 64, 72),
        "edge_channels": 20,
        "hypothesis": "baseline reference",
    },
    {
        "id": "E0",
        "name": "early_emphasis",
        "channels": (44, 44, 48, 52, 64, 72),
        "edge_channels": 20,
        "hypothesis": "extra capacity near high-resolution early features",
    },
    {
        "id": "L0",
        "name": "low_mid_stage2_emphasis",
        "channels": (24, 48, 52, 52, 64, 72),
        "edge_channels": 20,
        "hypothesis": "extra low-mid detail capacity including EFE low source",
    },
    {
        "id": "M0",
        "name": "middle_stage3_emphasis",
        "channels": (16, 32, 56, 56, 72, 72),
        "edge_channels": 20,
        "hypothesis": "extra middle feature capacity",
    },
    {
        "id": "D0",
        "name": "mid_deep_stage4_emphasis",
        "channels": (12, 24, 44, 68, 76, 80),
        "edge_channels": 20,
        "hypothesis": "extra capacity at transition into deep PV-Mamba stages",
    },
    {
        "id": "P0",
        "name": "stage5_emphasis",
        "channels": (12, 20, 36, 56, 84, 80),
        "edge_channels": 20,
        "hypothesis": "extra late bottleneck and decoder-adjacent capacity",
    },
    {
        "id": "T0",
        "name": "late_stage6_emphasis",
        "channels": (12, 20, 32, 48, 80, 96),
        "edge_channels": 20,
        "hypothesis": "extra deepest semantic capacity",
    },
    {
        "id": "X1",
        "name": "edge_width_high",
        "channels": (12, 20, 32, 44, 64, 72),
        "edge_channels": 96,
        "hypothesis": "wider EFE hidden edge branch",
    },
    {
        "id": "F1",
        "name": "efe_source_emphasis",
        "channels": (12, 48, 40, 48, 64, 96),
        "edge_channels": 20,
        "hypothesis": "extra capacity in EFE source stages f2 and f6",
    },
    {
        "id": "F2",
        "name": "efe_source_plus_edge",
        "channels": (12, 40, 36, 48, 64, 88),
        "edge_channels": 40,
        "hypothesis": "joint test of EFE source features and wider edge branch",
    },
]


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
    return Path(__file__).resolve().parents[2]


def parse_float_list(raw: str) -> list[float]:
    values = ast.literal_eval("[" + raw + "]")
    return [float(value) for value in values]


def parse_metrics_text(text: str) -> ParsedMetrics:
    epoch_means: list[float] = []
    epoch_times: list[float] = []
    for line in text.splitlines():
        dice_match = PSEUDO_DICE_RE.search(line)
        if dice_match:
            dice_values = parse_float_list(dice_match.group(1))
            if dice_values:
                epoch_means.append(sum(dice_values) / len(dice_values))
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


def count_params(channels: tuple[int, ...], edge_channels: int) -> int:
    from nnunetv2.nets.LMUNet import LMUNet

    model = LMUNet(in_channels=1, out_channels=4, channels=channels, edge_channels=edge_channels)
    return sum(parameter.numel() for parameter in model.parameters())


def latest_training_log(results_root: Path) -> Path | None:
    logs = list(results_root.rglob("training_log*.txt"))
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def run_experiment(
    experiment: dict,
    args: argparse.Namespace,
    package_root: Path,
    output_root: Path,
) -> dict:
    exp_label = f"{experiment['id']}_{experiment['name']}"
    exp_results_root = output_root / exp_label
    if args.overwrite and exp_results_root.exists():
        shutil.rmtree(exp_results_root)
    exp_results_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(args.nnunet_raw)
    env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    env["nnUNet_results"] = str(exp_results_root)
    env["LMUNET_OUTPUT_FOLDER"] = str(exp_results_root / "fold_0")
    env["LMUNET_CHANNELS"] = ",".join(str(channel) for channel in experiment["channels"])
    env["LMUNET_EDGE_CHANNELS"] = str(experiment["edge_channels"])
    env["LMUNET_EPOCHS"] = str(args.epochs)
    env["LMUNET_BATCH_SIZE"] = str(args.batch_size)
    env["LMUNET_DISABLE_CHECKPOINTING"] = "0" if args.keep_checkpoints else "1"
    env["LMUNET_SKIP_FINAL_VALIDATION"] = "0" if args.run_final_validation else "1"
    env["LMUNET_SKIP_ARCH_PLOT"] = "0" if args.plot_architecture else "1"
    env["LMUNET_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    if args.iters is not None:
        env["LMUNET_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters is not None:
        env["LMUNET_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    params = count_params(experiment["channels"], experiment["edge_channels"])
    command = [
        "nnUNetv2_train",
        str(args.dataset),
        args.configuration,
        str(args.fold),
        "-tr",
        "nnUNetTrainerLMUNet",
    ]

    print(f"\n=== {experiment['id']} {experiment['name']} ===")
    print("channels:", experiment["channels"])
    print("edge_channels:", experiment["edge_channels"])
    print("params:", params)
    print("results:", exp_results_root)
    print("command:", " ".join(command))

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=package_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration_s = time.perf_counter() - start
    print(completed.stdout)
    (exp_results_root / "subprocess_output.txt").write_text(completed.stdout, errors="replace")

    log_file = latest_training_log(exp_results_root)
    metrics = parse_training_log(log_file) if log_file is not None else ParsedMetrics([], None, None, None, None, [])
    if not metrics.pseudo_dice:
        metrics = parse_metrics_text(completed.stdout)
    mean_epoch_time = sum(metrics.epoch_times) / len(metrics.epoch_times) if metrics.epoch_times else None

    return {
        "id": experiment["id"],
        "name": experiment["name"],
        "channels": ",".join(str(channel) for channel in experiment["channels"]),
        "edge_channels": experiment["edge_channels"],
        "params": params,
        "params_m": params / 1e6,
        "hypothesis": experiment["hypothesis"],
        "returncode": completed.returncode,
        "duration_s": duration_s,
        "mean_epoch_time_s": mean_epoch_time,
        "final_dice": metrics.final_dice,
        "best_dice": metrics.best_dice,
        "momentum": metrics.momentum,
        "acceleration": metrics.acceleration,
        "num_dice_epochs": len(metrics.pseudo_dice),
        "pseudo_dice_by_epoch": ";".join(f"{value:.6f}" for value in metrics.pseudo_dice),
        "log_file": str(log_file) if log_file else "",
        "results_root": str(exp_results_root),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "name",
        "channels",
        "edge_channels",
        "params",
        "params_m",
        "final_dice",
        "best_dice",
        "momentum",
        "acceleration",
        "num_dice_epochs",
        "mean_epoch_time_s",
        "duration_s",
        "returncode",
        "hypothesis",
        "pseudo_dice_by_epoch",
        "log_file",
        "results_root",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows: list[dict], output_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib could not be imported: {exc}")
        return

    plotted = [row for row in rows if row["final_dice"] is not None]
    if not plotted:
        print("Skipping plots because no final Dice values were parsed.")
        return

    x = [row["params_m"] for row in plotted]
    y = [row["final_dice"] for row in plotted]
    labels = [row["id"] for row in plotted]
    momentum = [row["momentum"] if row["momentum"] is not None else 0.0 for row in plotted]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("LM-UNet Pareto: final Dice vs parameters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(x, y, c=momentum, cmap="coolwarm")
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("LM-UNet Pareto: color = momentum")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label="Momentum")
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice_momentum.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run LM-UNet Pareto experiments.")
    parser.add_argument("--configs", nargs="+", default=[experiment["id"] for experiment in EXPERIMENTS])
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--val-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--plot-architecture", action="store_true")
    parser.add_argument(
        "--run-final-validation",
        action="store_true",
        help="Run nnU-Net's final full validation after training. Disabled by default for sweeps.",
    )
    parser.add_argument("--output-root", type=Path, default=root / "data" / "nnUNET_results_pareto" / "lmunet_pareto")
    parser.add_argument("--nnunet-raw", type=Path, default=root / "data" / "nnUNet_raw")
    parser.add_argument("--nnunet-preprocessed", type=Path, default=root / "data" / "nnUNet_preprocessed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    selected = {experiment["id"]: experiment for experiment in EXPERIMENTS}
    unknown = [config for config in args.configs if config not in selected]
    if unknown:
        raise ValueError(f"Unknown config IDs: {unknown}")

    rows = []
    for config_id in args.configs:
        row = run_experiment(selected[config_id], args, package_root, args.output_root)
        rows.append(row)
        write_csv(rows, args.output_root / "summary.csv")
        plot_results(rows, args.output_root)
        if row["returncode"] != 0:
            print(f"Experiment {config_id} failed with return code {row['returncode']}. Stopping.")
            return row["returncode"]

    print("\n=== Summary ===")
    for row in rows:
        print(
            f"{row['id']:>3} params={row['params_m']:.3f}M "
            f"final_dice={row['final_dice']} momentum={row['momentum']} "
            f"duration_s={row['duration_s']:.1f}"
        )
    print("CSV:", args.output_root / "summary.csv")
    print("Plot:", args.output_root / "pareto_final_dice.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
