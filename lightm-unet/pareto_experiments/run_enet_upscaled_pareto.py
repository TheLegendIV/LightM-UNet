"""Pareto sweep for ENetUpscaled — 15-epoch architectural search on ARCADE."""
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
# channels = (ch0, ch1, ch2, ch3, ch4)
#   ch0   initial block (H/2)
#   ch1   stage-1 encoder (H/4, 4 blocks)
#   ch2   stage-2 flat encoder (H/4, 12 blocks, no stride)
#   ch3   stage-3 encoder (H/8, 12 blocks)
#   ch4   stage-4 deep encoder (H/16, 4 blocks)
# Decoder channels are fixed by MaxUnpool constraint:
#   H/8 decoder = ch3, H/4 decoder = ch2, H/2 decoder = ch0
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "id": "EU0",
        "name": "baseline",
        "channels": (20, 72, 72, 144, 256),
        "hypothesis": "ENetUpscaled reference: all five changes, original ENet channel widths",
    },
    {
        "id": "EU1",
        "name": "narrow",
        "channels": (16, 56, 56, 112, 192),
        "hypothesis": "Narrow budget — test whether architecture improvement beats width",
    },
    {
        "id": "EU2",
        "name": "wide_context",
        "channels": (20, 72, 128, 192, 256),
        "hypothesis": "Wider stage-2 flat (ch2=128) + stage-3 (ch3=192) context at H/4 and H/8",
    },
    {
        "id": "EU3",
        "name": "deep_stage4",
        "channels": (20, 72, 72, 144, 320),
        "hypothesis": "Extra capacity at the new H/16 stage (ch4=320)",
    },
    {
        "id": "EU4",
        "name": "shallow_wide",
        "channels": (20, 96, 96, 144, 256),
        "hypothesis": "Wider stages 1+2 (ch1=ch2=96) — tests if the extra blocks benefit from more width",
    },
    {
        "id": "EU5",
        "name": "stage3_stage4_heavy",
        "channels": (20, 72, 72, 196, 288),
        "hypothesis": "Deep encoder emphasis: extra capacity in stage-3 (ch3=196) and stage-4 (ch4=288)",
    },
    {
        "id": "EU6",
        "name": "fully_scaled",
        "channels": (24, 96, 96, 192, 256),
        "hypothesis": "Uniformly wider network — global capacity increase",
    },
    # --- Additional vessel-aware configs for 3-class ARCADE segmentation ---
    {
        "id": "EU7",
        "name": "highres_flat",
        "channels": (24, 96, 144, 144, 256),
        "hypothesis": (
            "Wide flat H/4 stage (ch2=144) preserves high-resolution detail "
            "for thin vessel segmentation before downsampling"
        ),
    },
    {
        "id": "EU8",
        "name": "narrow_deep",
        "channels": (16, 48, 48, 192, 288),
        "hypothesis": (
            "Narrow early stages, heavy bottleneck — concentrate capacity where "
            "long-range context discriminates vessel classes"
        ),
    },
    {
        "id": "EU9",
        "name": "asym_stage3",
        "channels": (20, 72, 72, 144, 256),
        "asym_stage3": True,
        "hypothesis": (
            "Same channels as EU0 but stage-3 extra blocks use asym-heavy pattern "
            "(reg, asym, asym, dil4) — tests vessel-oriented 5×1/1×5 kernels vs dil8"
        ),
    },
]


def count_params(channels: tuple[int, ...], asym_stage3: bool = False) -> int:
    from nnunetv2.nets.ENetUpscaled import ENetUpscaled

    model = ENetUpscaled(in_channels=1, out_channels=4, channels=channels, asym_heavy_stage3=asym_stage3)
    return sum(p.numel() for p in model.parameters())


def experiment_label(exp: dict) -> str:
    return f"{exp['id']}_{exp['name']}"


def attach_param_budgets(experiments: list[dict]) -> list[dict]:
    baseline_params = count_params(experiments[0]["channels"])
    result = []
    for exp in experiments:
        row = dict(exp)
        params = count_params(row["channels"], row.get("asym_stage3", False))
        row["params"] = params
        row["params_m"] = params / 1e6
        row["param_increase_pct"] = ((params / baseline_params) - 1.0) * 100.0
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
    env["ENETUSC_OUTPUT_FOLDER"] = str(exp_results_root / "fold_0")
    env["ENETUSC_CHANNELS"] = ",".join(str(c) for c in exp["channels"])
    env["ENETUSC_ASYM_STAGE3"] = "1" if exp.get("asym_stage3", False) else "0"
    env["ENETUSC_EPOCHS"] = str(args.epochs)
    env["ENETUSC_BATCH_SIZE"] = str(args.batch_size)
    env["ENETUSC_DISABLE_CHECKPOINTING"] = "0" if args.keep_checkpoints else "1"
    env["ENETUSC_SKIP_FINAL_VALIDATION"] = "0" if args.run_final_validation else "1"
    env["ENETUSC_SKIP_ARCH_PLOT"] = "0" if args.plot_architecture else "1"
    env["ENETUSC_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    if args.iters is not None:
        env["ENETUSC_ITERATIONS_PER_EPOCH"] = str(args.iters)
    if args.val_iters is not None:
        env["ENETUSC_VAL_ITERATIONS_PER_EPOCH"] = str(args.val_iters)

    command = [
        "nnUNetv2_train",
        str(args.dataset),
        args.configuration,
        str(args.fold),
        "-tr",
        "nnUNetTrainerENetUpscaled",
    ]

    print(f"\n=== {exp['id']} {exp['name']} ===")
    print("channels:", exp["channels"])
    print(f"params:   {exp['params_m']:.3f}M  ({exp['param_increase_pct']:+.1f}% vs EU0)")
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
        "name": exp["name"],
        "channels": ",".join(str(c) for c in exp["channels"]),
        "params": exp["params"],
        "params_m": exp["params_m"],
        "param_increase_pct": exp["param_increase_pct"],
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
        "id", "name", "channels", "params", "params_m", "param_increase_pct",
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

    momentum = [r["momentum"] if r["momentum"] is not None else 0.0 for r in plotted]

    plt.figure(figsize=(9, 5))
    sc = plt.scatter(
        [r["params_m"] for r in plotted],
        [r["final_dice"] for r in plotted],
        c=momentum,
        cmap="coolwarm",
        s=80,
    )
    for r in plotted:
        plt.annotate(r["id"], (r["params_m"], r["final_dice"]),
                     textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Parameters (M)")
    plt.ylabel("Final mean pseudo Dice")
    plt.title("ENetUpscaled Pareto sweep — color = momentum")
    plt.colorbar(sc, label="Momentum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "pareto_final_dice.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="ENetUpscaled Pareto sweep (default 15 epochs).")
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
        default=root / "data" / "nnUNET_results_pareto" / "enet_upscaled_e15",
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

    print("=== ENetUpscaled configs ===")
    for cid in args.configs:
        e = by_id[cid]
        print(
            f"  {e['id']:>3}  params={e['params_m']:.3f}M  "
            f"({e['param_increase_pct']:+.1f}% vs EU0)  "
            f"channels={e['channels']}"
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
            f"  {row['id']:>3}  params={row['params_m']:.3f}M  "
            f"final_dice={row['final_dice']}  duration_s={row['duration_s']:.0f}"
        )
    print("CSV:", args.output_root / "summary.csv")
    print("Plot:", args.output_root / "pareto_final_dice.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
