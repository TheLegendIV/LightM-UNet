"""Builds nnUNetTrainerENet_8_2_relu_no_reg_fullwidth's own real HAWQ
per-stage-quantized CombinedQuantENet from its trained FP32 checkpoint, then
CALIBRATES the newly-introduced Brevitas quantizers on real data before
saving -- same rationale/mechanism as calibrate_8_2_relu_no_reg_d2_
projected.py (confirmed load-bearing there: loss noisy/non-monotonic
without calibration, clean and monotonically decreasing with it).

Output checkpoint is meant as the WARM START for
nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth's own real QAT
fine-tuning (ENET_PRETRAINED_CHECKPOINT), not a final deployable artifact on
its own -- see compression/slurm/qat_8_2_relu_no_reg_fullwidth_25ep.job.

Usage:
    python compression/post-quantization/calibrate_8_2_relu_no_reg_fullwidth.py \\
        --stage-bits-file compression/hawq/stage_bits_8_2_relu_no_reg_fullwidth_minres.json \\
        --out-net-name nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth_calibrated
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from brevitas.graph.calibrate import calibration_mode

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet, block_names_for, expand_stage_bits  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}


def load_calibration_batches(dataset_name: str, n_images: int, seed: int = 0) -> list[torch.Tensor]:
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not image_files:
        raise FileNotFoundError(f"No preprocessed .npy images found under {preprocessed_dir}")
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_images, len(image_files)))
    return [torch.from_numpy(np.load(p)).float() for p in sampled]


def calibrate(quant_model: torch.nn.Module, calibration_batches: list[torch.Tensor], device: str) -> int:
    quant_model.to(device)
    quant_model.train()
    n_used = 0
    with torch.no_grad(), calibration_mode(quant_model):
        for batch in calibration_batches:
            try:
                quant_model(batch.to(device))
                n_used += 1
            except RuntimeError as error:
                print(f"  [skip] calibration image with shape {tuple(batch.shape)} failed forward pass: {error}")
    quant_model.eval()
    if n_used == 0:
        raise RuntimeError("Every calibration image failed -- can't calibrate quantizer scales at all.")
    return n_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-net-name", default="nnUNetTrainerENet_8_2_relu_no_reg_fullwidth")
    parser.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--out-net-name", required=True)
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--stage-bits-file", required=True, type=Path)
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-calibration-images", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.stage_bits_file) as f:
        stage_bits = json.load(f)
    block_names = block_names_for(BOTTLENECKS_PER_STAGE)
    block_weight_bits, block_act_bits = expand_stage_bits(
        stage_bits["stage_weight_bits"], stage_bits["stage_act_bits"], STAGE_MODULE_ATTRS, block_names,
    )

    source_model_folder = NNUNET_RESULTS / args.dataset_name / f"{args.source_net_name}__{args.plans_name}__{args.configuration}"
    source_checkpoint_path = source_model_folder / f"fold_{args.fold}" / args.source_checkpoint_name
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint_path}")

    print(f"Loading source FP32 checkpoint: {source_checkpoint_path}")
    quant_model = CombinedQuantENet.from_pretrained(
        source_checkpoint_path, block_weight_bits, block_act_bits,
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=True, separable_dilated=False, trainable_slope=False,
    )

    print(f"Calibrating on real preprocessed images (device={args.device})...")
    calibration_batches = load_calibration_batches(args.dataset_name, args.n_calibration_images, seed=args.calibration_seed)
    n_used = calibrate(quant_model, calibration_batches, args.device)
    print(f"Calibration used {n_used}/{len(calibration_batches)} images.")
    quant_model.to("cpu")

    out_model_folder = NNUNET_RESULTS / args.dataset_name / f"{args.out_net_name}__{args.plans_name}__{args.configuration}"
    out_fold_dir = out_model_folder / f"fold_{args.fold}"
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    for meta_file in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        src = source_model_folder / meta_file
        if src.exists():
            (out_model_folder / meta_file).write_bytes(src.read_bytes())

    reference_checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    network_weights = dict(quant_model.state_dict())
    network_weights.update(dict(quant_model.named_parameters(remove_duplicate=False)))
    new_checkpoint = dict(reference_checkpoint)
    new_checkpoint["network_weights"] = network_weights
    new_checkpoint["trainer_name"] = "nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth"
    out_checkpoint_path = out_fold_dir / "checkpoint_best.pth"
    torch.save(new_checkpoint, out_checkpoint_path)
    print(f"Saved calibrated checkpoint: {out_checkpoint_path}")


if __name__ == "__main__":
    main()
