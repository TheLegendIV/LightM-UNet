"""Run inference if needed, compute dice/clDice/n_components + params/FLOPs
for one trained ENet checkpoint, and write/update its row in results.csv.

The general-purpose successor to analysis/501_ARCADE/record_architecture_stats.py
(architecture stats only, two-baseline scope) -- this covers every stage's
sweep configs, reusing the same counting (compression/utils.py) and the same
topology/dice primitives (analysis/501_ARCADE/segmentation_topology.py), not
reimplementing either.

Usage:
    python compression/collect_results.py \
        --net-name nnUNetTrainerENet_E1 --stage stage1 \
        --channels 20,72,144,72,20

    python compression/collect_results.py \
        --net-name stage1b_no_dilated --stage stage1b \
        --channels 20,72,144,72,20 --use-dilated 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import fcntl  # POSIX only (HPC/Linux) -- guarded for local Windows dev
except ImportError:
    fcntl = None

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
ANALYSIS_ROOT = REPO_ROOT / "analysis" / "501_ARCADE"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(ANALYSIS_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.nets.QuantENet import QuantENet  # noqa: E402
import segmentation_topology as topo  # noqa: E402
from utils import count_bops, count_flops, count_params  # noqa: E402

DATASET_NAME = "Dataset501_ARCADE"
DATASET_ID = "501"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"
NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
IMAGES_TS_DIR = NNUNET_RAW / DATASET_NAME / "imagesTs"
LABELS_TS_DIR = NNUNET_RAW / DATASET_NAME / "labelsTs"
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

RESULTS_COLUMNS = [
    "config_name", "stage", "f_i", "f1", "f2", "f3", "f4", "f5",
    "bottlenecks_per_stage", "decoder_type", "ops_flags", "quant_bits",
    "params", "flops", "bops", "dice", "cldice", "n_components",
    "epochs", "converged_flag", "seed",
]


def parse_tuple5(value: str, name: str) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 5:
        raise ValueError(f"--{name} must have exactly 5 comma-separated integers, got {value!r}.")
    return parts


def parse_channels(value: str) -> tuple[int, ...]:
    """5 values (initial, stage1, stage2/3, stage4, stage5) or 6 if
    stage2/stage3 are split (initial, stage1, stage2, stage3, stage4,
    stage5) -- see ENet.py's 6-tuple channels form (upscale/'s max_unpool
    track)."""
    parts = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) not in (5, 6):
        raise ValueError(f"--channels must have 5 or 6 comma-separated integers, got {value!r}.")
    return parts


def run_inference(
    net_name: str,
    channels: tuple[int, ...],
    bottlenecks_per_stage: tuple[int, ...],
    decoder_type: str,
    use_dilated: bool,
    use_asymmetric: bool,
    use_strided: bool,
    configuration: str,
    plans_name: str,
    fold: int,
    checkpoint_name: str,
    device: str,
    quant_bits: int = 32,
    use_dsc: bool = False,
    context_pattern: str = "default",
    use_prelu: bool = True,
) -> Path:
    """Uses `nnUNetv2_predict_from_modelfolder` (-m <exact folder>), NOT
    plain `nnUNetv2_predict` (-tr/-p/-c/-d): the latter's folder resolution
    (get_output_folder) depends only on trainer_class/plans/configuration,
    identical across every sweep config -- it can't distinguish
    nnUNetTrainerENet_E1 from nnUNetTrainerENet_stage1b_no_dilated, both
    trained via the same trainer_class with ENET_OUTPUT_FOLDER redirecting
    each to its own net_name-suffixed folder (matches
    train_enet_e1.job / record_architecture_stats.py's convention, which
    DOES use net_name for the folder path). -m sidesteps that ambiguity by
    pointing at the fold-containing folder directly."""
    prediction_dir = NNUNET_RAW / DATASET_NAME / f"labelsPr_{net_name}"
    model_folder = NNUNET_RESULTS / DATASET_NAME / f"{net_name}__{plans_name}__{configuration}"
    checkpoint_path = model_folder / f"fold_{fold}" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not IMAGES_TS_DIR.exists():
        raise FileNotFoundError(f"imagesTs not found: {IMAGES_TS_DIR}")

    prediction_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    env["ENET_CHANNELS"] = ",".join(str(c) for c in channels)
    env["ENET_BOTTLENECKS"] = ",".join(str(n) for n in bottlenecks_per_stage)
    env["ENET_DECODER_TYPE"] = decoder_type
    env["ENET_USE_DILATED"] = "1" if use_dilated else "0"
    env["ENET_USE_ASYMMETRIC"] = "1" if use_asymmetric else "0"
    env["ENET_USE_STRIDED"] = "1" if use_strided else "0"
    env["ENET_USE_DSC"] = "1" if use_dsc else "0"
    env["ENET_CONTEXT_PATTERN"] = context_pattern
    env["ENET_USE_PRELU"] = "1" if use_prelu else "0"
    if quant_bits != 32:
        # Picked up by nnUNetTrainerENetQuant.build_network_architecture,
        # dynamically imported via the checkpoint's own stored trainer_name
        # (see nnUNetPredictor.initialize_from_trained_model_folder) -- not
        # passed as a CLI flag, same as every other ENET_* knob.
        env["ENET_QUANT_BITS"] = str(quant_bits)

    command = [
        shutil.which("nnUNetv2_predict_from_modelfolder") or "nnUNetv2_predict_from_modelfolder",
        "-i", str(IMAGES_TS_DIR),
        "-o", str(prediction_dir),
        "-m", str(model_folder),
        "-f", str(fold),
        "-chk", checkpoint_name,
        "-device", device,
        "--disable_progress_bar",
        "--disable_tta",
    ]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=PACKAGE_ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"nnUNetv2_predict failed with return code {completed.returncode}")
    return prediction_dir


def compute_eval_metrics(prediction_dir: Path) -> dict:
    """Mean dice / clDice / predicted-component-count across every matched
    labelsTs/labelsPr_{net_name} case pair -- reuses
    segmentation_topology.dice_score / cldice_score / fragmentation_stats,
    not a separate reimplementation."""
    dices, cldices, n_components_list = [], [], []
    for _, gt, pred in topo.iter_matched_cases(LABELS_TS_DIR, prediction_dir):
        gt_fg = gt > topo.BACKGROUND
        pred_fg = pred > topo.BACKGROUND
        dices.append(topo.dice_score(gt_fg, pred_fg))
        cldices.append(topo.cldice_score(gt_fg, pred_fg))
        n_components_list.append(topo.fragmentation_stats(gt, pred)["pred_components"])
    if not dices:
        raise FileNotFoundError(
            f"No matched labelsTs/{prediction_dir.name} case pairs -- check predictions exist."
        )
    return {
        "dice": sum(dices) / len(dices),
        "cldice": sum(cldices) / len(cldices),
        "n_components": sum(n_components_list) / len(n_components_list),
    }


def read_training_info(fold_dir: Path, checkpoint_name: str, trailing_window: int = 15) -> dict:
    """epochs/converged_flag read from the checkpoint's OWN stored metadata
    -- NOT debug.json, which turned out (checked against a real checkpoint,
    not assumed) to be a static snapshot written once at trainer
    initialization: it always shows current_epoch=0 regardless of how much
    training actually happened, silently wrong for every real run.

    converged_flag follows agent_instructions_1.yaml's actual definition
    ("false if val Dice still rising at the end"), not "did it reach
    num_epochs" -- compares the mean EMA foreground Dice over the training
    run's last two `trailing_window`-epoch halves (needs
    checkpoint_final.pth specifically for the full curve, since
    checkpoint_best.pth's own log only runs up to whichever epoch was
    best)."""
    checkpoint_path = fold_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"No {checkpoint_name} at {fold_dir} -- epochs/converged_flag left blank.")
        return {"epochs": None, "converged_flag": None}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epochs = checkpoint.get("current_epoch")

    final_checkpoint_path = fold_dir / "checkpoint_final.pth"
    converged_flag = None
    if final_checkpoint_path.exists():
        final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
        ema_dice = [float(x) for x in final_checkpoint.get("logging", {}).get("ema_fg_dice", [])]
        if len(ema_dice) >= trailing_window * 2:
            early_mean = sum(ema_dice[-trailing_window * 2:-trailing_window]) / trailing_window
            late_mean = sum(ema_dice[-trailing_window:]) / trailing_window
            converged_flag = late_mean <= early_mean  # still net rising -> NOT converged
        else:
            print(f"Only {len(ema_dice)} logged epochs in checkpoint_final.pth -- "
                  f"too few for a reliable trend (need >= {trailing_window * 2}), converged_flag left blank.")
    else:
        print(f"No checkpoint_final.pth at {fold_dir} -- converged_flag left blank (need the full loss curve).")

    return {"epochs": epochs, "converged_flag": converged_flag}


def upsert_row(row: dict) -> None:
    """Read-modify-write on the shared results.csv -- multiple Slurm array
    tasks (e.g. upscale/graduate.py's graduation array, or any of
    compression/slurm/*_array.job's cells) can call this within seconds of
    each other. Without a lock, two tasks reading the same pre-update state
    both add their own new row and the later write clobbers the earlier
    one's -- silently losing a whole run's result. flock serializes the
    whole read+merge+write critical section so every writer's read reflects
    every prior writer's completed write."""
    row = {col: row.get(col) for col in RESULTS_COLUMNS}
    lock_path = RESULTS_CSV.parent / ".results.csv.lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if RESULTS_CSV.exists():
                existing = pd.read_csv(RESULTS_CSV)
                existing = existing[existing["config_name"] != row["config_name"]]
                combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            else:
                combined = pd.DataFrame([row], columns=RESULTS_COLUMNS)
            combined.to_csv(RESULTS_CSV, index=False)
            print(f"Wrote {RESULTS_CSV} ({len(combined)} rows).")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-name", required=True, help="config_name, and the labelsPr_<net-name> prediction folder.")
    parser.add_argument("--trainer-class", default="nnUNetTrainerENet", help="Informational only (not used for inference -- see run_inference's docstring). Kept for parity with the training job's -tr flag.")
    parser.add_argument("--stage", required=True, help="e.g. stage1, stage1b, stage2, early_probe.")
    parser.add_argument("--channels", required=True, type=parse_channels)
    parser.add_argument("--bottlenecks", default="4,8,8,2,1", type=lambda v: parse_tuple5(v, "bottlenecks"))
    parser.add_argument("--decoder-type", default="max_unpool", choices=["max_unpool", "upsample_conv"])
    parser.add_argument("--use-dilated", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-asymmetric", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-strided", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-dsc", type=int, default=0, choices=[0, 1], help="Depthwise-separable inner conv (rejects combination with --use-asymmetric 1).")
    parser.add_argument("--context-pattern", default="default", choices=["default", "sparse"],
                         help="'sparse' = regular/dilated4/regular/dilated16 (section 2a's div2/div4 bottleneck axis), no 2/8 rungs, never asymmetric.")
    parser.add_argument("--use-prelu", type=int, default=1, choices=[0, 1], help="0 = collapse the encoder's PReLU to plain ReLU too (section 1d's ablation) -- decoder is always ReLU regardless, see ENet.py.")
    parser.add_argument("--quant-bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-inference", action="store_true", help="Assume labelsPr_<net-name> already exists.")
    args = parser.parse_args()

    # FLOPs/MACs always come from the plain FP32 ENet: thop silently
    # undercounts a QuantENet by ~40x (doesn't recognize Brevitas's quant
    # layers as countable ops -- measured, see utils.count_bops's
    # docstring), and MAC count is topology-determined, not quantization-
    # determined (verified topology-identical between ENet/QuantENet).
    fp32_model = ENet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=args.channels,
        bottlenecks_per_stage=args.bottlenecks,
        decoder_type=args.decoder_type,
        use_dilated=bool(args.use_dilated),
        use_asymmetric=bool(args.use_asymmetric),
        use_strided=bool(args.use_strided),
        use_dsc=bool(args.use_dsc),
        context_pattern=args.context_pattern,
        use_prelu=bool(args.use_prelu),
    )
    macs, flops = count_flops(fp32_model, args.in_channels, tuple(args.input_hw))

    if args.quant_bits == 32:
        total_params, _ = count_params(fp32_model)
        bops = None
    else:
        # Real deployed model's param count (close to but not identical to
        # the FP32 count -- Brevitas quant layers carry a little extra
        # bookkeeping, e.g. scale factors).
        quant_model = QuantENet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=args.channels,
            bottlenecks_per_stage=args.bottlenecks,
            decoder_type=args.decoder_type,
            use_dilated=bool(args.use_dilated),
            use_asymmetric=bool(args.use_asymmetric),
            use_strided=bool(args.use_strided),
            use_dsc=bool(args.use_dsc),
            weight_bit_width=args.quant_bits,
            act_bit_width=args.quant_bits,
        )
        total_params, _ = count_params(quant_model)
        bops = count_bops(macs, args.quant_bits)

    prediction_dir = NNUNET_RAW / DATASET_NAME / f"labelsPr_{args.net_name}"
    if not args.skip_inference and not prediction_dir.exists():
        prediction_dir = run_inference(
            net_name=args.net_name,
            channels=args.channels,
            bottlenecks_per_stage=args.bottlenecks,
            decoder_type=args.decoder_type,
            use_dilated=bool(args.use_dilated),
            use_asymmetric=bool(args.use_asymmetric),
            use_strided=bool(args.use_strided),
            configuration=args.configuration,
            plans_name=args.plans_name,
            fold=args.fold,
            checkpoint_name=args.checkpoint_name,
            device=args.device,
            quant_bits=args.quant_bits,
            use_dsc=bool(args.use_dsc),
            context_pattern=args.context_pattern,
            use_prelu=bool(args.use_prelu),
        )
    eval_metrics = compute_eval_metrics(prediction_dir)

    fold_dir = (
        NNUNET_RESULTS / DATASET_NAME / f"{args.net_name}__{args.plans_name}__{args.configuration}"
        / f"fold_{args.fold}"
    )
    training_info = read_training_info(fold_dir, args.checkpoint_name)

    # args.channels is ENet.py's 5-value tuple (initial, stage1, stage23,
    # stage4, stage5) -- stage2/stage3 share one value architecturally --
    # or the 6-value split form (initial, stage1, stage2, stage3, stage4,
    # stage5) used by upscale/'s max_unpool track. results.csv's schema
    # always has separate f2/f3 columns (equal in the 5-value case).
    if len(args.channels) == 5:
        f_i, f1, f23, f4, f5 = args.channels
        f2, f3 = f23, f23
    else:
        f_i, f1, f2, f3, f4, f5 = args.channels
    row = {
        "config_name": args.net_name,
        "stage": args.stage,
        "f_i": f_i, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
        "bottlenecks_per_stage": ",".join(str(n) for n in args.bottlenecks),
        "decoder_type": args.decoder_type,
        "ops_flags": f"dilated={args.use_dilated},asymmetric={args.use_asymmetric},strided={args.use_strided},dsc={args.use_dsc},context_pattern={args.context_pattern},prelu={args.use_prelu}",
        "quant_bits": args.quant_bits,
        "params": total_params,
        "flops": flops,
        "bops": bops,
        "seed": args.seed,
        **eval_metrics,
        **training_info,
    }
    upsert_row(row)
    for key, value in row.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
