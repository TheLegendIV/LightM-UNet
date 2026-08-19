"""Ad-hoc Dice check for the HAWQ-searched QuantENet23_1 scheme
(compression/hawq/stage_bits_23_1.json) against the FP32 23_1 baseline, on
the real held-out fold_0 validation split (data/nnUNet_preprocessed/.../
splits_final.json, 200 cases).

Calibrates the quantized model's Brevitas quantizer scales first (same
calibration_mode pattern as compression/post-quantization/ptq.py's
calibrate() -- from_pretrained() only transfers conv/BN weights, it does
NOT calibrate quantizer scales, so skipping this step gives meaningless
Dice numbers).

NOT the same protocol compression/collect_results.py's official numbers use
(that goes through nnU-Net's real nnUNetv2_predict inference pipeline --
mirroring/TTA, resampling back to original case resolution, etc.). This is
a direct patch-level hard-Dice check on the already-512x512 preprocessed
val patches, no TTA -- a quick, honest read on relative FP32-vs-quantized
degradation, not a drop-in replacement for the official validation number.

Usage:
    python compression/hawq/eval_dice.py --n-val-cases 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.nets.QuantENet23_1 import QuantENet23_1  # noqa: E402

from brevitas.graph.calibrate import calibration_mode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, IN_CHANNELS,
    OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, USE_ASYMMETRIC,
)

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"
NET_NAME = "nnUNetTrainerENet_23_1_s19_warmstart_4c"


def load_case(case_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans_2d"
    img = torch.from_numpy(np.load(d / f"{case_id}.npy")).float()
    seg = torch.from_numpy(np.load(d / f"{case_id}_seg.npy")).long()
    seg[seg == -1] = 0
    return img, seg


def hard_dice_per_class(pred: torch.Tensor, gt: torch.Tensor, n_classes: int) -> np.ndarray:
    """pred, gt: (H, W) integer label maps. Returns per-class Dice (nan
    where a class has zero pixels in both pred and gt for this case)."""
    dice = np.full(n_classes, np.nan)
    for c in range(n_classes):
        p = (pred == c)
        g = (gt == c)
        union = p.sum().item() + g.sum().item()
        if union == 0:
            continue
        inter = (p & g).sum().item()
        dice[c] = 2 * inter / union
    return dice


def evaluate(model: torch.nn.Module, case_ids: list[str], device: str, n_classes: int) -> dict:
    model.eval()
    all_dice = []
    n_skipped = 0
    with torch.no_grad():
        for case_id in case_ids:
            img, seg = load_case(case_id)
            img = img.to(device)
            try:
                out = model(img)
            except RuntimeError:
                n_skipped += 1
                continue
            out_t = out.value if hasattr(out, "value") else out
            pred = out_t.argmax(dim=1).squeeze(0).cpu()
            gt = seg.squeeze(0).squeeze(0)
            all_dice.append(hard_dice_per_class(pred, gt, n_classes))
    all_dice = np.stack(all_dice)  # (n_cases, n_classes)
    per_class_mean = np.nanmean(all_dice, axis=0)
    fg_mean = np.nanmean(per_class_mean[1:])  # exclude background (class 0)
    return {
        "n_cases_used": len(all_dice), "n_skipped": n_skipped,
        "per_class_dice": per_class_mean.tolist(), "mean_foreground_dice": float(fg_mean),
    }


def build_fp32(checkpoint_name: str) -> ENet:
    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        decoder_type=DECODER_TYPE, use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    ckpt_path = NNUNET_RESULTS / DATASET_NAME / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0" / checkpoint_name
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    return model


def calibrate_quant_model(model: QuantENet23_1, n_images: int, device: str, seed: int = 0) -> int:
    """Same pattern as compression/post-quantization/ptq.py's calibrate():
    forward-only passes over real preprocessed TRAINING images (not val --
    calibration must not see validation data) so every quantizer's observer
    collects real activation statistics before finalizing scale/zero-point."""
    import random
    d = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans_2d"
    train_files = sorted(p for p in d.glob("train_*.npy") if not p.name.endswith("_seg.npy"))
    rng = random.Random(seed)
    sampled = rng.sample(train_files, k=min(n_images, len(train_files)))
    model.to(device)
    model.train()
    n_used = 0
    with torch.no_grad(), calibration_mode(model):
        for p in sampled:
            img = torch.from_numpy(np.load(p)).float().to(device)
            try:
                model(img)
                n_used += 1
            except RuntimeError:
                continue
    model.eval()
    return n_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-bits-file", type=Path, default=Path("compression/hawq/stage_bits_23_1.json"))
    parser.add_argument("--slope-map-file", type=Path, default=Path("compression/post-quantization/slope_maps/23_1_s19_warmstart_4c.json"))
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--n-val-cases", type=int, default=200)
    parser.add_argument("--n-calibration-images", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "splits_final.json") as f:
        splits = json.load(f)
    val_cases = splits[0]["val"][: args.n_val_cases]
    print(f"Evaluating on {len(val_cases)} real held-out fold_0 validation cases.")

    print("\n=== FP32 baseline (23_1, no quantization) ===")
    fp32 = build_fp32(args.checkpoint_name).to(args.device)
    fp32_result = evaluate(fp32, val_cases, args.device, OUT_CHANNELS)
    print(json.dumps(fp32_result, indent=2))

    print("\n=== HAWQ-searched QuantENet23_1 (calibrated) ===")
    with open(args.stage_bits_file) as f:
        bits = json.load(f)
    with open(args.slope_map_file) as f:
        slope_map = json.load(f)
    ckpt_path = NNUNET_RESULTS / DATASET_NAME / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0" / args.checkpoint_name
    quant = QuantENet23_1.from_pretrained(ckpt_path, bits["stage_weight_bits"], bits["stage_act_bits"], slope_map)
    n_calib = calibrate_quant_model(quant, args.n_calibration_images, args.device)
    print(f"Calibrated on {n_calib}/{args.n_calibration_images} real training images.")
    quant_result = evaluate(quant, val_cases, args.device, OUT_CHANNELS)
    print(json.dumps(quant_result, indent=2))

    print("\n=== Summary ===")
    print(f"FP32 mean foreground Dice:  {fp32_result['mean_foreground_dice']:.4f}")
    print(f"HAWQ-quant mean fg Dice:    {quant_result['mean_foreground_dice']:.4f}")
    print(f"Delta:                      {quant_result['mean_foreground_dice'] - fp32_result['mean_foreground_dice']:+.4f}")


if __name__ == "__main__":
    main()
