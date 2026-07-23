"""Graduate winners from the upscale/ pareto sweep to full-length training.

Cherry-picked from wip_pc's own "graduation" step, which had no dedicated
script -- it just re-invoked the pareto runner with `--configs <winner_id>
--epochs 150` and checkpointing turned back on, writing into the standard
nnUNet_results tree instead of the throwaway pareto sweep root. This script
makes that explicit: pick winners off upscale/results/pareto_e15/summary.csv
(and pareto_final_dice.png) by eye, then run

    python upscale/graduate.py --configs MU05 UC03

Each graduated run:
  - reuses the SAME channels/bottlenecks from run_upscale_pareto.py's
    EXPERIMENTS table (no re-typing, no drift between the sweep and the
    graduation run it's supposedly reproducing)
  - trains nnUNetTrainerENet for the full epoch count (default 150, matching
    the rest of compression/'s runs) with checkpointing + final validation
    ENABLED (the sweep disables both for speed)
  - writes into the standard nnUNet_results/{dataset}/{trainer}__{plans}__{config}
    layout under run name `upscale_<id>`, exactly like a normal
    compression/slurm/*.job run
  - calls compression/collect_results.py at the end, so the graduated result
    lands in compression/results.csv (stage=upscale_graduate) alongside
    Original/O2/O4/... for direct comparison -- if a graduated config beats
    Original, IT becomes the new baseline the 37-run compression plan prunes
    from, not Original.

Usage:
    python upscale/graduate.py --configs MU05 UC03
    python upscale/graduate.py --configs MU05 --epochs 150 --skip-collect
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pareto_common import ensure_preprocessed, repo_root
from run_upscale_pareto import (
    EXPERIMENTS, USE_ASYMMETRIC, USE_DILATED, USE_DSC, USE_STRIDED,
)

REPO_ROOT = repo_root()

DATASET_ID = "501"
DATASET_NAME = "Dataset501_ARCADE"
TRAINER_CLASS = "nnUNetTrainerENet"


def graduate_one(exp: dict, args: argparse.Namespace) -> int:
    run_name = f"upscale_{exp['id']}"
    output_folder = (
        args.nnunet_results / DATASET_NAME
        / f"{TRAINER_CLASS}_{run_name}__nnUNetPlans__2d" / f"fold_{args.fold}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(args.nnunet_raw)
    env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    env["nnUNet_results"] = str(args.nnunet_results)
    env["ENET_CHANNELS"] = ",".join(str(c) for c in exp["channels"])
    env["ENET_BOTTLENECKS"] = ",".join(str(c) for c in exp["bottlenecks"])
    env["ENET_DECODER_TYPE"] = exp["decoder_type"]
    env["ENET_USE_DILATED"] = "1" if USE_DILATED else "0"
    env["ENET_USE_ASYMMETRIC"] = "1" if USE_ASYMMETRIC else "0"
    env["ENET_USE_STRIDED"] = "1" if USE_STRIDED else "0"
    env["ENET_USE_DSC"] = "1" if USE_DSC else "0"
    env["ENET_EPOCHS"] = str(args.epochs)
    env["ENET_SEED"] = str(args.seed)
    env["PYTHONHASHSEED"] = str(args.seed)
    # Checkpointing + final validation stay ON (unlike the 15-epoch sweep) --
    # this is a real run whose checkpoint feeds collect_results.py / the
    # rest of compression/.

    command = ["nnUNetv2_train", DATASET_ID, "2d", str(args.fold), "-tr", TRAINER_CLASS]
    print(f"\n=== Graduating {exp['id']} ({exp['name']}) -> {run_name} ===")
    print("channels:", exp["channels"], "bottlenecks:", exp["bottlenecks"], "epochs:", args.epochs)
    print("output:", output_folder)

    if (output_folder / "checkpoint_final.pth").exists() or (output_folder / "checkpoint_best.pth").exists():
        print(f"Checkpoint already exists at {output_folder} -- skipping training.")
        returncode = 0
    else:
        completed = subprocess.run(command, cwd=str(REPO_ROOT / "enet"), env=env)
        returncode = completed.returncode
        if returncode != 0:
            print(f"Training failed for {exp['id']} (rc={returncode}).")
            return returncode

    if args.skip_collect:
        return returncode

    collect_command = [
        sys.executable, str(REPO_ROOT / "compression" / "collect_results.py"),
        "--net-name", f"{TRAINER_CLASS}_{run_name}",
        "--stage", "upscale_graduate",
        "--channels", ",".join(str(c) for c in exp["channels"]),
        "--bottlenecks", ",".join(str(c) for c in exp["bottlenecks"]),
        "--decoder-type", exp["decoder_type"],
        "--use-dilated", "1" if USE_DILATED else "0",
        "--use-asymmetric", "1" if USE_ASYMMETRIC else "0",
        "--use-strided", "1" if USE_STRIDED else "0",
        "--use-dsc", "1" if USE_DSC else "0",
        "--trainer-class", TRAINER_CLASS,
    ]
    collect_env = os.environ.copy()
    collect_env["nnUNet_raw"] = str(args.nnunet_raw)
    collect_env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    collect_env["nnUNet_results"] = str(args.nnunet_results)
    print("=== Collecting results ===")
    print(" ".join(collect_command))
    collected = subprocess.run(collect_command, cwd=str(REPO_ROOT), env=collect_env)
    return collected.returncode


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Graduate upscale pareto winners to full-length training.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment IDs from run_upscale_pareto.py's EXPERIMENTS table, e.g. MU05 UC03.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-collect", action="store_true", help="Train only, don't call compression/collect_results.py afterward.")
    parser.add_argument("--nnunet-raw", type=Path, default=root / "data" / "nnUNet_raw")
    parser.add_argument("--nnunet-preprocessed", type=Path, default=root / "data" / "nnUNet_preprocessed")
    parser.add_argument("--nnunet-results", type=Path, default=root / "data" / "nnUNet_results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    by_id = {e["id"]: e for e in EXPERIMENTS}
    unknown = [c for c in args.configs if c not in by_id]
    if unknown:
        raise ValueError(f"Unknown config IDs: {unknown}. Valid: {list(by_id)}")

    ensure_preprocessed("501", "Dataset501_ARCADE", args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results)

    for cid in args.configs:
        rc = graduate_one(by_id[cid], args)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
