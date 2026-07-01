"""Pareto sweep for ENet Global Context Search (G1–G11).

Each experiment adds or replaces components with a global context mechanism
relative to the ENet baseline. Fixed channels (20,72,144,72,20).
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
        "id": "G1",
        "name": "mamba_bridge",
        "hypothesis": "+1 PVMamba at bridge; minimal global context probe",
    },
    {
        "id": "G2",
        "name": "mamba_s3",
        "hypothesis": "Stage 3 replaced by 4 PVMamba (forward scan); Mamba as dilated-conv substitute",
    },
    {
        "id": "G3",
        "name": "mamba_s3_4dir",
        "hypothesis": "Stage 3 replaced by 4 directional PVMamba (0→1→2→3); multi-direction scan",
    },
    {
        "id": "G4",
        "name": "mamba_s23",
        "hypothesis": "Stages 2+3 both replaced by 4 PVMamba each; maximum Mamba coverage",
    },
    {
        "id": "G5",
        "name": "mamba_transitions",
        "hypothesis": "+1 PVMamba after down2 and +1 after stage 3; global context at transition points",
    },
    {
        "id": "G6",
        "name": "mamba_parallel_eca",
        "hypothesis": "Parallel Mamba+conv branches at stages 2+3, ECA fusion; additive global context",
    },
    {
        "id": "G7",
        "name": "mamba_decoder",
        "hypothesis": "+1 PVMamba at decoder H/4 (after up4); global context during reconstruction",
    },
    {
        "id": "G8",
        "name": "mha_bridge",
        "hypothesis": "+1 full MHA at bridge (O(N²)); pairwise vs Mamba sequential scan",
    },
    {
        "id": "G9",
        "name": "window_attn_bridge",
        "hypothesis": "+1 windowed MHA at bridge (8×8 windows); local pairwise attention",
    },
    {
        "id": "G10",
        "name": "cross_depth_mamba",
        "hypothesis": "Stages 2+3 concatenated into 8192-token sequence for Mamba; cross-depth context",
    },
    {
        "id": "G11",
        "name": "linear_attn_s1",
        "hypothesis": "+1 linear attention at stage 1 (H/4, 16384 tokens); high-res O(N) global context",
    },
]


def count_params(experiment_id: str) -> int:
    from nnunetv2.nets.ENetGlobalCtx import build_gctx_model
    model = build_gctx_model(experiment_id, in_channels=1, out_channels=4, channels=CHANNELS)
    return sum(p.numel() for p in model.parameters())


def experiment_label(exp: dict) -> str:
    return f"{exp['id']}_{exp['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    from nnunetv2.nets.ENet import ENet
    baseline_params = sum(p.numel() for p in ENet().parameters())
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
        "GCX_EXPERIMENT": exp["id"],
        "GCX_CHANNELS": ",".join(str(c) for c in CHANNELS),
        "GCX_EPOCHS": str(args.epochs),
        "GCX_BATCH_SIZE": str(args.batch_size),
        "GCX_OUTPUT_FOLDER": str(exp_out),
        "GCX_DISABLE_CHECKPOINTING": "0",
        "GCX_SKIP_FINAL_VALIDATION": "0",
        "GCX_SKIP_ARCH_PLOT": "1",
        "nnUNet_raw": env.get("nnUNet_raw", ""),
        "nnUNet_preprocessed": env.get("nnUNet_preprocessed", ""),
        "nnUNet_results": str(output_root),
    })
    if args.iters:
        env["GCX_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters:
        env["GCX_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    cmd = [
        "nnUNetv2_train", "501", "2d", "0",
        "-tr", "nnUNetTrainerENetGlobalCtx",
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
    print(f"\n{'='*80}")
    print(f"ENet Global Context Search — {epochs} epochs")
    print(f"{'='*80}")
    header = f"{'ID':<5} {'Name':<22} {'Params':>8} {'Δ%':>7}  {'BestDice':>9} {'FinalDice':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        bd = f"{r['best_dice']:.4f}" if r["best_dice"] is not None else "   —   "
        fd = f"{r['final_dice']:.4f}" if r["final_dice"] is not None else "   —   "
        delta = f"{r['param_delta_pct']:+.1f}%" if r["param_delta_pct"] is not None else "   —"
        print(f"{r['id']:<5} {r['name']:<22} {r['params_m']:>6.3f}M {delta:>7}  {bd:>9} {fd:>10}")
    print(f"{'='*80}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ENet Global Context pareto sweep")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--val-iters", type=int, default=None)
    p.add_argument("--configs", nargs="+", default=None,
                   metavar="ID", help="Subset of experiment IDs (default: all)")
    p.add_argument("--output-root", type=Path,
                   default=Path("../data/nnUNET_results_pareto/enet_global_ctx_e15"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print param counts and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    package_root = repo_root() / "lightm-unet"

    experiments = attach_param_budgets(EXPERIMENTS)
    by_id = {e["id"]: e for e in experiments}

    if args.dry_run:
        from nnunetv2.nets.ENet import ENet
        baseline = sum(p.numel() for p in ENet().parameters())
        print(f"\nENet Global Context Search — dry run  baseline={baseline/1e6:.3f}M")
        print(f"{'ID':<5} {'Name':<22} {'Params':>8}  {'Δ vs ENet':>10}  Hypothesis")
        print("-" * 92)
        for e in experiments:
            delta = f"{e['param_delta_pct']:+.1f}%"
            print(f"{e['id']:<5} {e['name']:<22} {e['params_m']:>6.3f}M  {delta:>10}  {e['hypothesis'][:52]}")
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
        print(f"params:  {exp['params_m']:.3f}M  ({exp['param_delta_pct']:+.1f}% vs ENet baseline)")
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
