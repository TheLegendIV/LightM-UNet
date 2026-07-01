"""Pareto sweep for ENetSkip — skip connection ablation on the ENet baseline."""
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


# ---------------------------------------------------------------------------
# Experiment table
# Channel schedule is fixed: (16, 64, 128, 64, 16)
#   initial : 16ch, H/2  (x_initial — potential 5x injection point)
#   stage1  : 64ch, H/4  (x_1x — potential 4x injection point)
#   stage23 : 128ch, H/8 (x_2x/x_3x — dilated context stages)
#   stage4  : 64ch, H/4  (x_4x, must equal stage1 for MaxUnpool)
#   stage5  : 16ch, H/2  (x_5x, must equal initial for MaxUnpool)
#
# Groups:
#   A — raw skip, Conv1×1 fusion (cheapest gate — is the skip useful at all?)
#   B — raw skip, fusion depth (Conv3×3 / Bottleneck — does fusion capacity matter?)
#   C — attention-gated skip, Conv1×1 (decoder-guided gating)
#   D — EFE+EFF edge-guided skip from LM-UNet (requires mamba-ssm; compute-heavy)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "id": "baseline",
        "group": "—",
        "name": "no_skip",
        "hypothesis": "ENet original, no skip connections (control)",
    },
    # --- Group A ---
    {
        "id": "A1",
        "group": "A",
        "name": "raw_4x",
        "hypothesis": "Raw skip x_1x→4x with Conv1×1 fusion; ablates 4x injection alone",
    },
    {
        "id": "A2",
        "group": "A",
        "name": "raw_5x",
        "hypothesis": "Raw skip x_initial→5x with Conv1×1 fusion; ablates 5x injection alone",
    },
    {
        "id": "A3",
        "group": "A",
        "name": "raw_both",
        "hypothesis": "Raw skips at both 4x and 5x with Conv1×1 fusion (A1+A2 combined)",
    },
    # --- Group B ---
    {
        "id": "B1",
        "group": "B",
        "name": "conv3x3_both",
        "hypothesis": "A3 variant: 3×3 fusion adds local spatial mixing on top of channel reduction",
    },
    {
        "id": "B2",
        "group": "B",
        "name": "bottleneck_both",
        "hypothesis": "A3 variant: bottleneck fusion maximises capacity for skip integration",
    },
    # --- Group C ---
    {
        "id": "C1",
        "group": "C",
        "name": "attn_4x",
        "hypothesis": "Attention-gated skip at 4x: decoder-conditioned spatial gate on x_1x",
    },
    {
        "id": "C2",
        "group": "C",
        "name": "attn_5x",
        "hypothesis": "Attention-gated skip at 5x: decoder-conditioned spatial gate on x_initial",
    },
    {
        "id": "C3",
        "group": "C",
        "name": "attn_both",
        "hypothesis": "Attention-gated skips at both 4x and 5x injection points",
    },
    # --- Group D: lightweight edge-guided skip (pure convolution, no Mamba) ---
    {
        "id": "D1",
        "group": "D",
        "name": "edge_4x",
        "hypothesis": "LightEdgePrior(x_1x, x_3x)→1ch edge map; LightEdgeGate gates x_1x at 4x",
    },
    {
        "id": "D2",
        "group": "D",
        "name": "edge_both",
        "hypothesis": "Lightweight edge-gated skips at both 4x and 5x",
    },
    {
        "id": "D3",
        "group": "D",
        "name": "edge_triple_5x",
        "hypothesis": "Edge-gate at 4x + triple concat at 5x: x_5x + gate(x_initial) + x_initial (48→16ch)",
    },
    # --- E: residual addition instead of concat ---
    {
        "id": "E1",
        "group": "E",
        "name": "residual_4x",
        "hypothesis": "Residual addition at 4x: Conv1×1(x_1x)→ch_s4 added to x_4x; no concat, no fusion block",
    },
    # --- F: structural baseline — bilinear decoder, no feature skips ---
    {
        "id": "F1",
        "group": "F",
        "name": "bilinear_no_skip",
        "hypothesis": "MaxUnpool replaced by bilinear upsampling, no feature skips; isolates MaxUnpool contribution vs baseline",
    },
]

# Fixed channel schedule — not varied across experiments
CHANNELS = (20, 72, 144, 72, 20)  # ENet.py defaults — baseline == ENetOriginal exactly


def count_params(experiment_id: str) -> int:
    from nnunetv2.nets.ENetSkip import ENetSkip
    model = ENetSkip(in_channels=1, out_channels=4, channels=CHANNELS, experiment=experiment_id)
    return sum(p.numel() for p in model.parameters())


def experiment_label(exp: dict) -> str:
    return f"{exp['id']}_{exp['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    baseline_params = count_params("baseline")
    result = []
    for exp in experiments:
        row = dict(exp)
        params = count_params(row["id"])
        row["params"] = params
        row["params_m"] = params / 1e6
        row["param_delta_pct"] = ((params / baseline_params) - 1.0) * 100.0
        result.append(row)
    return result


def reuse_result(exp: dict, source: Path, output_root: Path) -> dict:
    """Copy an existing result directory as this experiment's output without retraining."""
    exp_results_root = output_root / experiment_label(exp)
    if not exp_results_root.exists():
        print(f"\n=== {exp['id']} [{exp['group']}] {exp['name']} (reusing {source}) ===")
        shutil.copytree(source, exp_results_root)
    else:
        print(f"\n=== {exp['id']} [{exp['group']}] {exp['name']} (already at {exp_results_root}) ===")

    log_file = latest_training_log(exp_results_root)
    metrics = (
        parse_training_log(log_file)
        if log_file is not None
        else ParsedMetrics([], None, None, None, None, [])
    )
    mean_epoch_time = (
        sum(metrics.epoch_times) / len(metrics.epoch_times) if metrics.epoch_times else None
    )
    return {
        "id": exp["id"],
        "group": exp["group"],
        "name": exp["name"],
        "params": exp["params"],
        "params_m": exp["params_m"],
        "param_delta_pct": exp["param_delta_pct"],
        "hypothesis": exp["hypothesis"],
        "returncode": 0,
        "duration_s": 0.0,
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
    env["ENETSK_OUTPUT_FOLDER"] = str(exp_results_root / "fold_0")
    env["ENETSK_EXPERIMENT"] = exp["id"]
    env["ENETSK_CHANNELS"] = ",".join(str(c) for c in CHANNELS)
    env["ENETSK_EPOCHS"] = str(args.epochs)
    env["ENETSK_BATCH_SIZE"] = str(args.batch_size)
    env["ENETSK_DISABLE_CHECKPOINTING"] = "0" if args.keep_checkpoints else "1"
    env["ENETSK_SKIP_FINAL_VALIDATION"] = "0" if args.run_final_validation else "1"
    env["ENETSK_SKIP_ARCH_PLOT"] = "0" if args.plot_architecture else "1"
    env["ENETSK_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    if args.iters is not None:
        env["ENETSK_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters is not None:
        env["ENETSK_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    command = [
        "nnUNetv2_train",
        str(args.dataset),
        args.configuration,
        str(args.fold),
        "-tr",
        "nnUNetTrainerENetSkip",
    ]

    print(f"\n=== {exp['id']} [{exp['group']}] {exp['name']} ===")
    print(f"params:  {exp['params_m']:.3f}M  ({exp['param_delta_pct']:+.1f}% vs baseline)")
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
        "id", "group", "name", "params", "params_m", "param_delta_pct",
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

    group_colors = {"—": "gray", "A": "steelblue", "B": "darkorange", "C": "green", "D": "purple", "E": "red", "F": "black"}

    plt.figure(figsize=(9, 5))
    for r in plotted:
        color = group_colors.get(r.get("group", "—"), "gray")
        plt.scatter(r["params_m"], r["final_dice"], color=color, s=80)
        plt.annotate(r["id"], (r["params_m"], r["final_dice"]),
                     textcoords="offset points", xytext=(5, 5))

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=f"Group {g}", markersize=8)
              for g, c in group_colors.items()]
    plt.legend(handles=legend, loc="lower right")
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("ENetSkip — skip connection ablation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="ENetSkip skip connection ablation sweep (default 15 epochs).")
    parser.add_argument("--configs", nargs="+", default=[e["id"] for e in EXPERIMENTS])
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument(
        "--reuse-baseline", type=Path, default=None, metavar="PATH",
        help="Copy this directory as the 'baseline' result instead of retraining it.",
    )
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
        default=root / "data" / "nnUNET_results_pareto" / "enet_skip_e15",
    )
    parser.add_argument("--nnunet-raw", type=Path, default=root / "data" / "nnUNet_raw")
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=root / "data" / "nnUNet_preprocessed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    experiments = attach_param_budgets(EXPERIMENTS)
    by_id = {e["id"]: e for e in experiments}

    unknown = [c for c in args.configs if c not in by_id]
    if unknown:
        raise ValueError(f"Unknown config IDs: {unknown}")

    print("=== ENetSkip configs ===")
    for cid in args.configs:
        e = by_id[cid]
        print(
            f"  {e['id']:>8}  [{e['group']}]  params={e['params_m']:.3f}M"
            f"  ({e['param_delta_pct']:+.1f}% vs baseline)  {e['name']}"
        )
    if args.dry_run:
        return 0

    rows = []
    for cid in args.configs:
        exp = by_id[cid]
        if cid == "baseline" and args.reuse_baseline is not None:
            row = reuse_result(exp, args.reuse_baseline, args.output_root)
        else:
            row = run_experiment(exp, args, package_root, args.output_root)
        rows.append(row)
        write_csv(rows, args.output_root / "summary.csv")
        plot_results(rows, args.output_root)
        if row["returncode"] != 0 and cid != "baseline":
            print(f"Experiment {cid} failed (rc={row['returncode']}). Stopping.")
            return row["returncode"]

    print("\n=== Summary ===")
    for row in rows:
        print(
            f"  {row['id']:>8}  [{row['group']}]  params={row['params_m']:.3f}M"
            f"  final_dice={row['final_dice']}  duration_s={row['duration_s']:.0f}"
        )
    print("CSV:", args.output_root / "summary.csv")
    print("Plot:", args.output_root / "pareto_final_dice.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
