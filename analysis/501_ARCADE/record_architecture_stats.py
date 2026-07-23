"""Record architecture stats (params/channels/bottlenecks/FLOPs) and training
hyperparameters/logs for a trained ENet checkpoint on Dataset501_ARCADE.

Doesn't need predictions or ground truth -- pure model/training-run
bookkeeping to sit alongside the dice/topology/shape-diagnostic CSVs, so
every {net_name} in results/ has one row of "what was this model, and how
was it trained" to go with "how well did it do."

Usage:
    python analysis/501_ARCADE/record_architecture_stats.py \
        --net-name nnUNetTrainerENet_E1 --channels 20,72,144,72,20
    python analysis/501_ARCADE/record_architecture_stats.py \
        --net-name nnUNetTrainerENet_Original --channels 16,64,128,64,16

Reads (if present, best-effort -- architecture stats alone don't need a
trained checkpoint, only the channel config; hyperparameters/logs do):
    $nnUNet_results/Dataset501_ARCADE/{net_name}__nnUNetPlans__2d/fold_0/
        debug.json, training_log_*.txt, progress.png

Writes:
    results/{net_name}_architecture_stats.csv   -- one row: params/channels/FLOPs/hyperparameters
    results/{net_name}_debug.json               -- copy of the trainer's debug.json (full dump)
    results/{net_name}_training_log.txt         -- copy of the most recent training log
    results/{net_name}_progress.png             -- copy of the loss-curve plot
    results/summary_architecture_stats.csv      -- combined across every net-name this has been run for
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compression"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from utils import count_flops, count_params  # noqa: E402

METRICS_DIR = Path(__file__).resolve().parent / "results"
DATASET_NAME = "Dataset501_ARCADE"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"

# A handful of the most relevant debug.json scalars, pulled into the flat CSV
# row for quick cross-model comparison -- the full dump is still copied
# verbatim as {net_name}_debug.json for anything else. debug.json stores
# every value as a plain str() (see nnUNetTrainer._save_debug_information),
# so these stay strings here too rather than being re-parsed/typed.
HYPERPARAM_KEYS = [
    "initial_lr", "weight_decay", "num_epochs", "num_iterations_per_epoch",
    "num_val_iterations_per_epoch", "current_epoch", "batch_size",
    "device", "torch_version", "gpu_name",
]


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(channels) != 5:
        raise ValueError("--channels must have exactly 5 comma-separated integers (matches ENET_CHANNELS).")
    return channels


def bottleneck_counts(model: ENet) -> dict[str, int]:
    """Number of bottleneck blocks per stage, read directly off the built
    module tree (not hardcoded), so this stays correct if ENet.py's stage
    depths ever change. regular5 became parametric (nn.Sequential) alongside
    bottlenecks_per_stage -- no longer assumed to always be exactly 1."""
    return {
        "n_bottlenecks_stage1": len(model.regular1),
        "n_bottlenecks_stage2": len(model.stage2),
        "n_bottlenecks_stage3": len(model.stage3),
        "n_bottlenecks_stage4": len(model.regular4),
        "n_bottlenecks_stage5": len(model.regular5),
    }


def read_debug_json(fold_dir: Path) -> dict:
    debug_path = fold_dir / "debug.json"
    if not debug_path.exists():
        return {}
    return json.loads(debug_path.read_text())


def copy_logs(fold_dir: Path, net_name: str) -> list[str]:
    """Copy debug.json / training_log_*.txt / progress.png into results/ so
    the raw training record sits next to the metrics CSVs, prefixed with
    net_name. Best-effort -- skips whatever isn't there yet (e.g. before a
    real training run has completed)."""
    copied = []
    for pattern, dest_name in [
        ("debug.json", f"{net_name}_debug.json"),
        ("progress.png", f"{net_name}_progress.png"),
    ]:
        src = fold_dir / pattern
        if src.exists():
            shutil.copy2(src, METRICS_DIR / dest_name)
            copied.append(dest_name)
    training_logs = sorted(fold_dir.glob("training_log_*.txt"))
    if training_logs:
        latest = max(training_logs, key=lambda p: p.stat().st_mtime)  # a resumed run can have several
        dest = METRICS_DIR / f"{net_name}_training_log.txt"
        shutil.copy2(latest, dest)
        copied.append(dest.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Record architecture/hyperparameter stats for a trained ENet checkpoint.")
    parser.add_argument("--net-name", required=True, help="Matches the trainer-folder prefix used in the .job scripts, e.g. nnUNetTrainerENet_E1.")
    parser.add_argument("--channels", required=True, type=parse_channels, help="Comma-separated 5 ints, matching ENET_CHANNELS, e.g. 20,72,144,72,20.")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2, help="2 for this dataset's binary background/vessel labeling (nnU-Net softmax convention: background + 1 foreground class).")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--results-dir", type=Path, default=NNUNET_RESULTS, help="Override if $nnUNet_results isn't REPO_ROOT/data/nnUNet_results.")
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    METRICS_DIR.mkdir(exist_ok=True)

    model = ENet(in_channels=args.in_channels, out_channels=args.out_channels, channels=args.channels)
    total_params, trainable_params = count_params(model)
    macs, flops = count_flops(model, args.in_channels, tuple(args.input_hw))

    initial_channels, stage1, stage23, stage4, stage5 = args.channels
    stats = {
        "net_name": args.net_name,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "initial_channels": initial_channels,
        "stage1_channels": stage1,
        "stage2_channels": stage23,
        "stage3_channels": stage23,
        "stage4_channels": stage4,
        "stage5_channels": stage5,
        **bottleneck_counts(model),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_h": args.input_hw[0],
        "input_w": args.input_hw[1],
        "macs": macs,
        "flops": flops,
        "gflops": (flops / 1e9) if flops is not None else None,
    }

    fold_dir = args.results_dir / DATASET_NAME / f"{args.net_name}__{args.plans_name}__{args.configuration}" / f"fold_{args.fold}"
    debug_info = read_debug_json(fold_dir)
    if debug_info:
        for key in HYPERPARAM_KEYS:
            if key in debug_info:
                stats[key] = debug_info[key]
    else:
        print(f"No debug.json found at {fold_dir} -- recording architecture stats only, no training hyperparameters.")

    copied = copy_logs(fold_dir, args.net_name) if fold_dir.exists() else []
    if copied:
        print(f"Copied logs: {', '.join(copied)}")

    stats_path = METRICS_DIR / f"{args.net_name}_architecture_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"\nWrote {stats_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Merge into the combined summary, replacing any prior row for this net_name.
    summary_path = METRICS_DIR / "summary_architecture_stats.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["net_name"] != args.net_name]
        combined = pd.concat([existing, pd.DataFrame([stats])], ignore_index=True)
    else:
        combined = pd.DataFrame([stats])
    combined = combined.sort_values("total_params", ignore_index=True)
    combined.to_csv(summary_path, index=False)
    print(f"Updated {summary_path} ({len(combined)} models)")


if __name__ == "__main__":
    main()
