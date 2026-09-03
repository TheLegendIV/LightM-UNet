"""Diagnostic (not a permanent pipeline stage): checks whether S19-joint's
and S5.6's per-block quantization scheme is ALREADY collapsed (near-zero
foreground Dice) immediately after warm-start + calibration, BEFORE any QAT
fine-tuning step ever runs.

Why this test: both configs show falling train_loss but Pseudo dice stuck
near 0 by epoch ~20-26 of real QAT. That's consistent with two very
different root causes that look identical from the training log alone:

  (A) The quantization scheme itself is broken at step 0 (bad per-block bit
      assignment, bad calibration/scale init, a slope-map mismatch) -- QAT
      then just fails to recover a badly-initialized starting point.
  (B) The quantization scheme is FINE at step 0 (comparable to FP32) but
      training dynamics destroy it within the first few epochs (LR too high
      for QAT fine-tuning, newly-trainable alpha parameters diverging,
      optimizer-state mismatch, etc.).

This script never trains anything -- it builds each Quant* model straight
from its own FP32 source checkpoint (from_pretrained), runs the same
forward-only calibration eval_dice.py's own calibrate_quant_model() uses,
and evaluates real held-out Dice immediately. A near-zero result here
means (A); a reasonable result (close to the FP32 source's own Dice) means
the collapse happens DURING training, i.e. (B).

Usage (run inside the dev container):
    python compression/hawq/diagnose_ptq_collapse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from nnunetv2.nets.QuantENet5_6Block import QuantENet5_6Block  # noqa: E402
from nnunetv2.nets.QuantENetS19Block import QuantENetS19Block  # noqa: E402

from brevitas.graph.calibrate import calibration_mode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dice import evaluate  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL_CASES = 60
N_CALIB_IMAGES = 64

# (label, fp32_net_name, block_bits_file, slope_map_file, quant_class,
#  fp32_channels/bottlenecks/etc -- only needed to load the plain FP32 model)
CONFIGS = [
    {
        "label": "S19-twopass (validated, control -- QAT'd to real dice=0.7458)",
        "fp32_net_name": "nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid",
        "block_bits_file": REPO_ROOT / "compression/hawq/artifacts/block_bits_s19_validated.json",
        "slope_map_file": REPO_ROOT / "compression/post-quantization/slope_maps/19_reginterleaved_separable_nonneg_block_double_mid.json",
        "quant_class": QuantENetS19Block,
        "fp32_kwargs": dict(
            channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 12, 12, 2, 1),
            decoder_type="upsample_conv", use_asymmetric=False,
            context_pattern="dense_dilation_reg_interleaved_double_mid",
            separable_dilated=True, use_prelu=True, prelu_variant="nonneg_block",
        ),
    },
    {
        "label": "S19-joint",
        "fp32_net_name": "nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid",
        "block_bits_file": REPO_ROOT / "compression/hawq/artifacts/block_bits_s19_joint.json",
        "slope_map_file": REPO_ROOT / "compression/post-quantization/slope_maps/19_reginterleaved_separable_nonneg_block_double_mid.json",
        "quant_class": QuantENetS19Block,
        "fp32_kwargs": dict(
            channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 12, 12, 2, 1),
            decoder_type="upsample_conv", use_asymmetric=False,
            context_pattern="dense_dilation_reg_interleaved_double_mid",
            separable_dilated=True, use_prelu=True, prelu_variant="nonneg_block",
        ),
    },
    {
        "label": "S5.6",
        "fp32_net_name": "nnUNetTrainerENet_5_6_separable_dense_dilation",
        "block_bits_file": REPO_ROOT / "compression/hawq/artifacts/block_bits_5_6.json",
        "slope_map_file": REPO_ROOT / "compression/post-quantization/slope_maps/5_6_separable_dense_dilation.json",
        "quant_class": QuantENet5_6Block,
        "fp32_kwargs": dict(
            channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
            decoder_type="upsample_conv", use_asymmetric=False,
            context_pattern="dense_dilation", separable_dilated=True,
            use_prelu=True, prelu_variant="standard",
        ),
    },
]
IN_CHANNELS = 1
OUT_CHANNELS = 5


def build_fp32(net_name: str, kwargs: dict, checkpoint_name: str = "checkpoint_final.pth") -> ENet:
    model = ENet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, **kwargs)
    ckpt_path = NNUNET_RESULTS / DATASET_NAME / f"{net_name}__nnUNetPlans__2d" / "fold_0" / checkpoint_name
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["network_weights"], strict=True)
    return model, ckpt_path


def calibrate(model: torch.nn.Module, n_images: int, device: str, seed: int = 0) -> int:
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
            import numpy as np
            img = torch.from_numpy(np.load(p)).float().to(device)
            try:
                model(img)
                n_used += 1
            except RuntimeError:
                continue
    model.eval()
    return n_used


def main() -> None:
    with open(NNUNET_PREPROCESSED / DATASET_NAME / "splits_final.json") as f:
        splits = json.load(f)
    val_cases = splits[0]["val"][:N_VAL_CASES]
    print(f"Using {len(val_cases)} real held-out fold_0 validation cases.\n")

    for cfg in CONFIGS:
        print(f"=== {cfg['label']} ===")
        fp32, ckpt_path = build_fp32(cfg["fp32_net_name"], cfg["fp32_kwargs"])
        fp32.to(DEVICE)
        fp32_result = evaluate(fp32, val_cases, DEVICE, OUT_CHANNELS)
        print(f"  FP32 source ({ckpt_path.name}) mean fg Dice:      {fp32_result['mean_foreground_dice']:.4f}")

        with open(cfg["block_bits_file"]) as f:
            bits = json.load(f)
        with open(cfg["slope_map_file"]) as f:
            slope_map = json.load(f)
        quant = cfg["quant_class"].from_pretrained(
            ckpt_path, bits["stage_weight_bits"], bits["stage_act_bits"], slope_map,
        )
        n_calib = calibrate(quant, N_CALIB_IMAGES, DEVICE)
        quant_result = evaluate(quant, val_cases, DEVICE, OUT_CHANNELS)
        print(f"  Quant (calibrated on {n_calib}/{N_CALIB_IMAGES}, NO QAT) mean fg Dice: {quant_result['mean_foreground_dice']:.4f}")
        print(f"  Delta vs FP32 source: {quant_result['mean_foreground_dice'] - fp32_result['mean_foreground_dice']:+.4f}")
        print()


if __name__ == "__main__":
    main()
