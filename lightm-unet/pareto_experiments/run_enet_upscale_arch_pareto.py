"""Pareto sweep for ENet Upscale Architectural Search (UAS0–UAS8).

Each experiment isolates one structural modification vs ENetOriginal (UAS0).
Fixed channels (20,72,144,72,20) for all experiments — variation is architectural,
not channel-width.
"""
from __future__ import annotations

import argparse
import csv
import os
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
        "id": "UAS0",
        "name": "baseline",
        "hypothesis": "ENetOriginal — 2-stage encoder, control for all modifications",
    },
    {
        "id": "UAS1",
        "name": "deep_h16",
        "hypothesis": "3rd downsampling to H/16 + 8-block context + decoder stage; deeper hierarchy for vessel topology",
    },
    {
        "id": "UAS2",
        "name": "extra_ctx",
        "hypothesis": "+2 bottlenecks in stages 2+3 (8→10 each); more depth at fixed H/8 resolution",
    },
    {
        "id": "UAS3",
        "name": "flat_h4",
        "hypothesis": "Stage 2 flat at H/4 (delayed stride); preserves resolution through semantic aggregation",
    },
    {
        "id": "UAS4",
        "name": "flat_h4_reldil",
        "hypothesis": "UAS3 + dilation rates 4/8/16/32; compensates proportional receptive field lost by resolution change",
    },
    {
        "id": "UAS5",
        "name": "double_cycle",
        "hypothesis": "Double context cycle in stages 2+3 (8→16 blocks); second pass through full dilation sequence",
    },
    {
        "id": "UAS6",
        "name": "learnable_ds",
        "hypothesis": "Strided 2×2 conv replaces MaxPool in both downsampling bottlenecks; learnable spatial aggregation",
    },
    {
        "id": "UAS7",
        "name": "wide_bottle",
        "hypothesis": "Stages 2+3 internal_ratio=2 (ch/2 vs ch/4); wider bottleneck compression for thin-structure features",
    },
    {
        "id": "UAS8",
        "name": "asym_s1",
        "hypothesis": "Stage 1 uses asymmetric 5×1/1×5 convs at H/4; anisotropic bias for elongated vessel structures",
    },
]


def count_params(experiment_id: str) -> int:
    from nnunetv2.nets.ENetUpscaleArch import build_uas_model
    model = build_uas_model(experiment_id, in_channels=1, out_channels=4, channels=CHANNELS)
    return sum(p.numel() for p in model.parameters())


def experiment_label(exp: dict) -> str:
    return f"{exp['id']}_{exp['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    baseline_params = count_params("UAS0")
    result = []
    for exp in experiments:
        row = dict(exp)
        params = count_params(row["id"])
        row["params"] = params
        row["params_m"] = params / 1e6
        row["param_delta_pct"] = ((params / baseline_params) - 1.0) * 100.0
        result.append(row)
    return result


def run_experiment(exp: dict, args: argparse.Namespace, package_root: Path,
                   output_root: Path) -> dict:
    label = experiment_label(exp)
    exp_out = output_root / label / "fold_0"
    exp_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "UAS_EXPERIMENT": exp["id"],
        "UAS_CHANNELS": ",".join(str(c) for c in CHANNELS),
        "UAS_EPOCHS": str(args.epochs),
        "UAS_BATCH_SIZE": str(args.batch_size),
        "UAS_OUTPUT_FOLDER": str(exp_out),
        "UAS_DISABLE_CHECKPOINTING": "0",
        "UAS_SKIP_FINAL_VALIDATION": "0",
        "UAS_SKIP_ARCH_PLOT": "1",
        "nnUNet_raw": env.get("nnUNet_raw", ""),
        "nnUNet_preprocessed": env.get("nnUNet_preprocessed", ""),
        "nnUNet_results": str(output_root),
    })
    if args.iters:
        env["UAS_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters:
        env["UAS_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    cmd = [
        "nnUNetv2_train", "501", "2d", "0",
        "-tr", "nnUNetTrainerENetUpscaleArch",
    ]
    if args.overwrite:
        cmd.append("--c")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(package_root))
    duration = time.time() - t0

    log_file = latest_training_log(exp_out)
    metrics = parse_training_log(log_file) if log_file else ParsedMetrics([], None, None, None, None, [])
    mean_epoch_time = (
        sum(metrics.epoch_times) / len(metrics.epoch_times) if metrics.epoch_times else None
    )

    return {
        "id": exp["id"],
        "name": exp["name"],
        "params": exp.get("params", 0),
        "params_m": exp.get("params_m", 0.0),
        "param_delta_pct": exp.get("param_delta_pct", 0.0),
        "best_dice": metrics.best_dice,
        "final_dice": metrics.final_dice,
        "mean_epoch_time_s": mean_epoch_time,
        "total_time_s": duration,
        "returncode": result.returncode,
        "hypothesis": exp.get("hypothesis", ""),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "name", "params", "params_m", "param_delta_pct",
        "best_dice", "final_dice",
        "mean_epoch_time_s", "total_time_s", "returncode", "hypothesis",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], output_root: Path, epochs: int) -> None:
    print(f"\n{'='*78}")
    print(f"ENet Upscale Architectural Search — {epochs} epochs")
    print(f"{'='*78}")
    header = f"{'ID':<6} {'Name':<18} {'Params':>8} {'Δ%':>7}  {'BestDice':>9} {'FinalDice':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        bd = f"{r['best_dice']:.4f}" if r["best_dice"] is not None else "   —   "
        fd = f"{r['final_dice']:.4f}" if r["final_dice"] is not None else "   —   "
        delta = f"{r['param_delta_pct']:+.1f}%" if r["param_delta_pct"] is not None else "   —"
        print(f"{r['id']:<6} {r['name']:<18} {r['params_m']:>6.3f}M {delta:>7}  {bd:>9} {fd:>10}")
    print(f"{'='*78}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ENet Upscale Architectural Search pareto sweep")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--val-iters", type=int, default=None)
    p.add_argument("--configs", nargs="+", default=None,
                   metavar="ID", help="Subset of experiment IDs to run (default: all)")
    p.add_argument("--output-root", type=Path,
                   default=Path("../data/nnUNET_results_pareto/enet_upscale_arch_e15"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print param counts and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    package_root = repo_root() / "lightm-unet"

    experiments = attach_param_budgets(EXPERIMENTS)
    by_id = {e["id"]: e for e in experiments}

    if args.dry_run:
        print(f"\nENet Upscale Arch Search — dry run (channels={CHANNELS})")
        print(f"{'ID':<6} {'Name':<18} {'Params':>8}  {'Δ vs UAS0':>10}  Hypothesis")
        print("-" * 90)
        for e in experiments:
            delta = f"{e['param_delta_pct']:+.1f}%"
            print(f"{e['id']:<6} {e['name']:<18} {e['params_m']:>6.3f}M  {delta:>10}  {e['hypothesis'][:55]}")
        return

    configs = args.configs if args.configs else [e["id"] for e in experiments]
    unknown = set(configs) - by_id.keys()
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. Valid: {list(by_id)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for cid in configs:
        exp = by_id[cid]
        print(f"\n=== {cid} [{exp['name']}] ===")
        print(f"params:  {exp['params_m']:.3f}M  ({exp['param_delta_pct']:+.1f}% vs UAS0)")
        print(f"results: {args.output_root / experiment_label(exp)}")
        row = run_experiment(exp, args, package_root, args.output_root)
        rows.append(row)
        write_csv(rows, args.output_root / "summary.csv")
        if row["returncode"] != 0:
            print(f"Experiment {cid} failed (rc={row['returncode']}). Stopping.")
            break

    write_summary(rows, args.output_root, args.epochs)
    print(f"CSV: {args.output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
