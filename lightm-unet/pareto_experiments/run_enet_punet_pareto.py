from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

from run_lmunet_pareto import (
    ParsedMetrics,
    latest_training_log,
    parse_metrics_text,
    parse_training_log,
    repo_root,
)


EXPERIMENTS = [
    {
        "id": "EN0",
        "name": "enet_baseline",
        "family": "enet",
        "trainer": "nnUNetTrainerENet",
        "channels": (16, 64, 128, 64, 16),
        "budget_group": "base_75pct",
        "hypothesis": "paper-faithful ENetOriginal baseline",
    },
    {
        "id": "EN1",
        "name": "enet_stage_balanced_small",
        "family": "enet",
        "trainer": "nnUNetTrainerENet",
        "channels": (20, 72, 144, 72, 20),
        "budget_group": "base_75pct",
        "hypothesis": "balanced ENet width increase below the main budget target",
    },
    {
        "id": "EN2",
        "name": "enet_stage_balanced_mid",
        "family": "enet",
        "trainer": "nnUNetTrainerENet",
        "channels": (20, 80, 160, 80, 20),
        "budget_group": "base_75pct",
        "hypothesis": "balanced ENet width increase around the middle of the budget",
    },
    {
        "id": "EN3",
        "name": "enet_context_heavy_target",
        "family": "enet",
        "trainer": "nnUNetTrainerENet",
        "channels": (16, 72, 176, 72, 16),
        "budget_group": "base_75pct",
        "hypothesis": "ENet capacity concentrated in the dilated context stages near +75 percent params",
    },
    {
        "id": "PU0",
        "name": "punet_baseline",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (16, 24, 36, 52, 72, 104),
        "budget_group": "base_75pct",
        "hypothesis": "PUNet baseline matched to the paper parameter count",
    },
    {
        "id": "PU1",
        "name": "punet_decoder_deep_small",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (16, 24, 36, 56, 88, 128),
        "budget_group": "base_75pct",
        "hypothesis": "PUNet extra capacity in deep/global and decoder-adjacent stages",
    },
    {
        "id": "PU2",
        "name": "punet_smooth_mid",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (20, 32, 44, 64, 96, 128),
        "budget_group": "base_75pct",
        "hypothesis": "PUNet smoother width scaling through middle stages",
    },
    {
        "id": "PU3",
        "name": "punet_target_75pct",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (20, 32, 48, 72, 104, 144),
        "budget_group": "base_75pct",
        "hypothesis": "PUNet near the shared +75 percent scaling budget",
    },
    {
        "id": "PU4",
        "name": "punet_stretch_110pct",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (24, 36, 52, 76, 112, 160),
        "budget_group": "punet_stretch",
        "hypothesis": "PUNet-only stretch run because baseline has fewer parameters",
    },
    {
        "id": "PU5",
        "name": "punet_stretch_117pct",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (24, 40, 56, 80, 112, 160),
        "budget_group": "punet_stretch",
        "hypothesis": "PUNet-only high scale run below the maximum stretch budget",
    },
    {
        "id": "PU6",
        "name": "punet_stretch_125pct",
        "family": "punet",
        "trainer": "nnUNetTrainerPUNet",
        "channels": (24, 40, 56, 80, 116, 164),
        "budget_group": "punet_stretch",
        "hypothesis": "PUNet-only maximum scale run at approximately +125 percent parameters",
    },
]


BASELINE_PARAMS: dict[str, int] = {}


def env_prefix(family: str) -> str:
    if family == "enet":
        return "ENET"
    if family == "punet":
        return "PUNET"
    raise ValueError(f"Unknown family: {family}")


def count_params(family: str, channels: tuple[int, ...]) -> int:
    if family == "enet":
        from nnunetv2.nets.ENet import ENet

        model = ENet(in_channels=1, out_channels=4, channels=channels)
    elif family == "punet":
        from nnunetv2.nets.PUNet import PUNet

        model = PUNet(in_channels=1, out_channels=4, channels=channels)
    else:
        raise ValueError(f"Unknown family: {family}")
    return sum(parameter.numel() for parameter in model.parameters())


def experiment_label(experiment: dict) -> str:
    return f"{experiment['id']}_{experiment['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    global BASELINE_PARAMS
    baseline_by_family = {
        experiment["family"]: count_params(experiment["family"], experiment["channels"])
        for experiment in experiments
        if experiment["id"].endswith("0")
    }
    BASELINE_PARAMS = baseline_by_family

    prepared = []
    for experiment in experiments:
        row = dict(experiment)
        params = count_params(row["family"], row["channels"])
        baseline = baseline_by_family[row["family"]]
        row["params"] = params
        row["params_m"] = params / 1e6
        row["param_increase_pct"] = ((params / baseline) - 1.0) * 100.0
        prepared.append(row)
    return prepared


def run_experiment(
    experiment: dict,
    args: argparse.Namespace,
    package_root: Path,
    output_root: Path,
) -> dict:
    exp_results_root = output_root / experiment_label(experiment)
    if args.overwrite and exp_results_root.exists():
        shutil.rmtree(exp_results_root)
    exp_results_root.mkdir(parents=True, exist_ok=True)

    prefix = env_prefix(experiment["family"])
    env = os.environ.copy()
    env["nnUNet_raw"] = str(args.nnunet_raw)
    env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    env["nnUNet_results"] = str(exp_results_root)
    env[f"{prefix}_OUTPUT_FOLDER"] = str(exp_results_root / "fold_0")
    env[f"{prefix}_CHANNELS"] = ",".join(str(channel) for channel in experiment["channels"])
    env[f"{prefix}_EPOCHS"] = str(args.epochs)
    env[f"{prefix}_BATCH_SIZE"] = str(args.batch_size)
    env[f"{prefix}_DISABLE_CHECKPOINTING"] = "0" if args.keep_checkpoints else "1"
    env[f"{prefix}_SKIP_FINAL_VALIDATION"] = "0" if args.run_final_validation else "1"
    env[f"{prefix}_SKIP_ARCH_PLOT"] = "0" if args.plot_architecture else "1"
    env[f"{prefix}_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    if args.iters is not None:
        env[f"{prefix}_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters is not None:
        env[f"{prefix}_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    command = [
        "nnUNetv2_train",
        str(args.dataset),
        args.configuration,
        str(args.fold),
        "-tr",
        experiment["trainer"],
    ]

    print(f"\n=== {experiment['id']} {experiment['name']} ===")
    print("family:", experiment["family"])
    print("channels:", experiment["channels"])
    print("params:", experiment["params"], f"({experiment['param_increase_pct']:.1f}% vs family baseline)")
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
        "family": experiment["family"],
        "trainer": experiment["trainer"],
        "budget_group": experiment["budget_group"],
        "channels": ",".join(str(channel) for channel in experiment["channels"]),
        "params": experiment["params"],
        "params_m": experiment["params_m"],
        "param_increase_pct": experiment["param_increase_pct"],
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
        "family",
        "trainer",
        "budget_group",
        "channels",
        "params",
        "params_m",
        "param_increase_pct",
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

    colors = {"enet": "tab:blue", "punet": "tab:orange"}
    plt.figure(figsize=(8, 5))
    for family in sorted({row["family"] for row in plotted}):
        family_rows = [row for row in plotted if row["family"] == family]
        plt.scatter(
            [row["params_m"] for row in family_rows],
            [row["final_dice"] for row in family_rows],
            label=family.upper(),
            color=colors.get(family),
        )
        for row in family_rows:
            plt.annotate(row["id"], (row["params_m"], row["final_dice"]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("ENetOriginal vs PUNet Pareto sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run ENetOriginal and PUNet Pareto experiments.")
    parser.add_argument("--configs", nargs="+", default=[experiment["id"] for experiment in EXPERIMENTS])
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--val-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--plot-architecture", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print selected configs and parameter budgets, then exit.")
    parser.add_argument(
        "--run-final-validation",
        action="store_true",
        help="Run nnU-Net's final full validation after training. Disabled by default for sweeps.",
    )
    parser.add_argument("--output-root", type=Path, default=root / "data" / "nnUNET_results_pareto" / "enet_punet_bs8_e10")
    parser.add_argument("--nnunet-raw", type=Path, default=root / "data" / "nnUNet_raw")
    parser.add_argument("--nnunet-preprocessed", type=Path, default=root / "data" / "nnUNet_preprocessed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    experiments = attach_param_budgets(EXPERIMENTS)
    selected = {experiment["id"]: experiment for experiment in experiments}
    unknown = [config for config in args.configs if config not in selected]
    if unknown:
        raise ValueError(f"Unknown config IDs: {unknown}")

    print("=== Selected configs ===")
    for config_id in args.configs:
        experiment = selected[config_id]
        print(
            f"{experiment['id']:>3} {experiment['family']:<5} "
            f"params={experiment['params_m']:.3f}M "
            f"increase={experiment['param_increase_pct']:.1f}% "
            f"channels={experiment['channels']}"
        )
    if args.dry_run:
        return 0

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
            f"{row['id']:>3} {row['family']:<5} params={row['params_m']:.3f}M "
            f"increase={row['param_increase_pct']:.1f}% final_dice={row['final_dice']} "
            f"duration_s={row['duration_s']:.1f}"
        )
    print("CSV:", args.output_root / "summary.csv")
    print("Plot:", args.output_root / "pareto_final_dice.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
