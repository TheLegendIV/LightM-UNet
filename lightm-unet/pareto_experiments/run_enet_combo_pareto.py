"""Pareto sweep for ENetCombo — cross product of the winning skip (E1/A3) and
context (G3/UAS2) ablations from ENetSkip and ENetGlobalCtx/ENetUpscaleArch.

See nnunetv2/nets/ENetCombo.py for the full rationale and the id -> (skip, ctx)
table. E1 and A3 both control the 4x skip injection point and can't be
combined with each other, so they run as two parallel families (EG*, AG*)
instead of one six-way cross product.
"""
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

CHANNELS = (20, 72, 144, 72, 20)

EXPERIMENTS = [
    {
        "id": "EG1",
        "group": "E",
        "skip": "E1",
        "ctx": "G3",
        "name": "e1_g3",
        "hypothesis": "E1 residual skip (4x) + G3 4-directional Mamba stage 3",
    },
    {
        "id": "EG2",
        "group": "E",
        "skip": "E1",
        "ctx": "UAS2",
        "name": "e1_uas2",
        "hypothesis": "E1 residual skip (4x) + UAS2 extra-depth stage 2 (10 blocks)",
    },
    {
        "id": "EG3",
        "group": "E",
        "skip": "E1",
        "ctx": "G3_UAS2",
        "name": "e1_g3_uas2",
        "hypothesis": "E1 residual skip (4x) + UAS2 stage 2 depth + G3 Mamba stage 3",
    },
    {
        "id": "AG1",
        "group": "A",
        "skip": "A3",
        "ctx": "G3",
        "name": "a3_g3",
        "hypothesis": "A3 raw skips (4x+5x) + G3 4-directional Mamba stage 3",
    },
    {
        "id": "AG2",
        "group": "A",
        "skip": "A3",
        "ctx": "UAS2",
        "name": "a3_uas2",
        "hypothesis": "A3 raw skips (4x+5x) + UAS2 extra-depth stage 2 (10 blocks)",
    },
    {
        "id": "AG3",
        "group": "A",
        "skip": "A3",
        "ctx": "G3_UAS2",
        "name": "a3_g3_uas2",
        "hypothesis": "A3 raw skips (4x+5x) + UAS2 stage 2 depth + G3 Mamba stage 3",
    },
]


def count_params(experiment_id: str) -> int:
    from nnunetv2.nets.ENetCombo import build_combo_model
    model = build_combo_model(experiment_id, in_channels=1, out_channels=4, channels=CHANNELS)
    return sum(p.numel() for p in model.parameters())


def experiment_label(exp: dict) -> str:
    return f"{exp['id']}_{exp['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    from nnunetv2.nets.ENet import ENet
    baseline_params = sum(p.numel() for p in ENet(in_channels=1, out_channels=4, channels=CHANNELS).parameters())
    result = []
    for exp in experiments:
        row = dict(exp)
        params = count_params(row["id"])
        row["params"] = params
        row["params_m"] = params / 1e6
        row["param_delta_pct"] = ((params / baseline_params) - 1.0) * 100.0
        result.append(row)
    return result


def run_experiment(
    exp: dict,
    args: argparse.Namespace,
    package_root: Path,
    output_root: Path,
) -> dict:
    exp_results_root = output_root / experiment_label(exp)
    if args.overwrite and exp_results_root.exists():
        shutil.rmtree(exp_results_root)
    exp_results_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(args.nnunet_raw)
    env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    env["nnUNet_results"] = str(exp_results_root)
    env["COMBO_OUTPUT_FOLDER"] = str(exp_results_root / "fold_0")
    env["COMBO_EXPERIMENT"] = exp["id"]
    env["COMBO_CHANNELS"] = ",".join(str(c) for c in CHANNELS)
    env["COMBO_EPOCHS"] = str(args.epochs)
    env["COMBO_BATCH_SIZE"] = str(args.batch_size)
    env["COMBO_DISABLE_CHECKPOINTING"] = "0" if args.keep_checkpoints else "1"
    env["COMBO_SKIP_FINAL_VALIDATION"] = "0" if args.run_final_validation else "1"
    env["COMBO_SKIP_ARCH_PLOT"] = "0" if args.plot_architecture else "1"
    env["COMBO_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    if args.iters is not None:
        env["COMBO_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters is not None:
        env["COMBO_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    command = [
        "nnUNetv2_train",
        str(args.dataset),
        args.configuration,
        str(args.fold),
        "-tr",
        "nnUNetTrainerENetCombo",
    ]

    print(f"\n=== {exp['id']} [{exp['group']}] {exp['name']} ===")
    print(f"skip={exp['skip']}  ctx={exp['ctx']}")
    print(f"params:  {exp['params_m']:.3f}M  ({exp['param_delta_pct']:+.1f}% vs ENet baseline)")
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
    metrics = (
        parse_training_log(log_file)
        if log_file is not None
        else ParsedMetrics([], None, None, None, None, [])
    )
    if not metrics.pseudo_dice:
        metrics = parse_metrics_text(completed.stdout)
    mean_epoch_time = (
        sum(metrics.epoch_times) / len(metrics.epoch_times) if metrics.epoch_times else None
    )

    return {
        "id": exp["id"],
        "group": exp["group"],
        "skip": exp["skip"],
        "ctx": exp["ctx"],
        "name": exp["name"],
        "params": exp["params"],
        "params_m": exp["params_m"],
        "param_delta_pct": exp["param_delta_pct"],
        "hypothesis": exp["hypothesis"],
        "returncode": completed.returncode,
        "duration_s": duration_s,
        "mean_epoch_time_s": mean_epoch_time,
        "final_dice": metrics.final_dice,
        "best_dice": metrics.best_dice,
        "momentum": metrics.momentum,
        "acceleration": metrics.acceleration,
        "num_dice_epochs": len(metrics.pseudo_dice),
        "pseudo_dice_by_epoch": ";".join(f"{v:.6f}" for v in metrics.pseudo_dice),
        "log_file": str(log_file) if log_file else "",
        "results_root": str(exp_results_root),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "group", "skip", "ctx", "name", "params", "params_m", "param_delta_pct",
        "final_dice", "best_dice", "momentum", "acceleration", "num_dice_epochs",
        "mean_epoch_time_s", "duration_s", "returncode", "hypothesis",
        "pseudo_dice_by_epoch", "log_file", "results_root",
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
        print(f"Skipping plots: {exc}")
        return

    plotted = [r for r in rows if r["final_dice"] is not None]
    if not plotted:
        return

    group_colors = {"E": "steelblue", "A": "darkorange"}
    group_labels = {"E": "E1 (residual 4x)", "A": "A3 (raw 4x+5x)"}

    plt.figure(figsize=(9, 5))
    for r in plotted:
        color = group_colors.get(r.get("group", "E"), "gray")
        plt.scatter(r["params_m"], r["final_dice"], color=color, s=90)
        plt.annotate(r["id"], (r["params_m"], r["final_dice"]),
                     textcoords="offset points", xytext=(5, 5))

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=group_labels[g], markersize=8)
              for g, c in group_colors.items()]
    plt.legend(handles=legend, loc="lower right")
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("ENetCombo — skip x context cross product")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice.png", dpi=160)
    plt.close()

    # Second plot: momentum-colored, mirrors run_enet_upscaled_pareto.py style
    momentum = [r["momentum"] if r["momentum"] is not None else 0.0 for r in plotted]
    plt.figure(figsize=(9, 5))
    sc = plt.scatter(
        [r["params_m"] for r in plotted],
        [r["final_dice"] for r in plotted],
        c=momentum,
        cmap="coolwarm",
        s=90,
    )
    for r in plotted:
        plt.annotate(r["id"], (r["params_m"], r["final_dice"]),
                     textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("ENetCombo Pareto sweep — color = momentum")
    plt.colorbar(sc, label="Momentum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "pareto_momentum.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="ENetCombo skip x context Pareto sweep.")
    parser.add_argument("--configs", nargs="+", default=[e["id"] for e in EXPERIMENTS])
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--val-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--plot-architecture", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs and parameter counts, then exit.")
    parser.add_argument("--run-final-validation", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data" / "nnUNET_results_pareto" / "enet_combo_e15",
    )
    parser.add_argument("--nnunet-raw", type=Path, default=root / "data" / "nnUNet_raw")
    parser.add_argument("--nnunet-preprocessed",
                        type=Path, default=root / "data" / "nnUNet_preprocessed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    experiments = attach_param_budgets(EXPERIMENTS)
    by_id = {e["id"]: e for e in experiments}

    unknown = [c for c in args.configs if c not in by_id]
    if unknown:
        raise ValueError(f"Unknown config IDs: {unknown}")

    print("=== ENetCombo configs ===")
    for cid in args.configs:
        e = by_id[cid]
        print(
            f"  {e['id']:>4}  [{e['group']}]  skip={e['skip']:<3} ctx={e['ctx']:<8}  "
            f"params={e['params_m']:.3f}M  ({e['param_delta_pct']:+.1f}% vs baseline)"
        )
    if args.dry_run:
        return 0

    rows = []
    for cid in args.configs:
        row = run_experiment(by_id[cid], args, package_root, args.output_root)
        rows.append(row)
        write_csv(rows, args.output_root / "summary.csv")
        plot_results(rows, args.output_root)
        if row["returncode"] != 0:
            print(f"Experiment {cid} failed (rc={row['returncode']}). Stopping.")
            return row["returncode"]

    print("\n=== Summary ===")
    for row in rows:
        print(
            f"  {row['id']:>4}  [{row['group']}]  params={row['params_m']:.3f}M  "
            f"final_dice={row['final_dice']}  momentum={row['momentum']}  duration_s={row['duration_s']:.0f}"
        )
    print("CSV:", args.output_root / "summary.csv")
    print("Plots:", args.output_root / "pareto_final_dice.png", "/", args.output_root / "pareto_momentum.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
