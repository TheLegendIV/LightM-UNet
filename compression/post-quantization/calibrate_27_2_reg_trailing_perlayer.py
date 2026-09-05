"""Builds nnUNetTrainerENet_27_2_reg_trailing's own real per-LAYER
HAWQ-quantized LayerQuantENet from its trained FP32 checkpoint, then
CALIBRATES the newly-introduced Brevitas quantizers on real data before
saving -- byte-for-byte the same mechanism as calibrate_12_separable_dense_
relu_perlayer.py, just for S27.2's own architecture shape (bottlenecks_per_
stage=(4,10,10,2,1), context_pattern="dense_dilation_reg_trailing" -- see
compression/hawq/config_27_2_reg_trailing.py).

Output checkpoint is meant as the WARM START for
nnUNetTrainerLayerQuantENet_27_2_reg_trailing_perlayer's own real QAT
fine-tuning (ENET_PRETRAINED_CHECKPOINT), not a final deployable artifact on
its own.

Usage:
    python compression/post-quantization/calibrate_27_2_reg_trailing_perlayer.py \\
        --layer-bits-file compression/hawq/artifacts/S27_2_ILP_outputs_perlayer/layer_bits_SITES_27_2_reg_trailing_joint_alpha0.5_candidatebits468_forcedsp70lut.json \\
        --out-net-name nnUNetTrainerLayerQuantENet_27_2_reg_trailing_joint_alpha0.5_candidatebits468_calibrated
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
from nnunetv2.nets.LayerQuantENet import LayerQuantENet  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 10, 10, 2, 1)
CONTEXT_PATTERN = "dense_dilation_reg_trailing"


def load_calibration_batches(dataset_name: str, n_images: int, seed: int = 0) -> list[torch.Tensor]:
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not image_files:
        raise FileNotFoundError(f"No preprocessed .npy images found under {preprocessed_dir}")
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_images, len(image_files)))
    return [torch.from_numpy(np.load(p)).float() for p in sampled]


def calibrate(quant_model: torch.nn.Module, calibration_batches: list[torch.Tensor], device: str, seed: int) -> int:
    """seed reseeds torch's GLOBAL RNG right before calibration starts -- see
    calibrate_12_separable_dense_relu_perlayer.py's own calibrate() docstring
    for the full rationale (Downsampling/UpsamplingBottleneck's nn.Dropout2d
    is active during calibration_mode, so without this the calibrated
    residual_add/out_act scales are corrupted by whatever the ambient global
    RNG state happens to be -- non-reproducible run-to-run even given
    identical calibration images)."""
    torch.manual_seed(seed)
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
    parser.add_argument("--source-net-name", default="nnUNetTrainerENet_27_2_reg_trailing")
    parser.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--out-net-name", required=True)
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--layer-bits-file", required=True, type=Path,
                         help="compression/hawq/expand_layer_bits.py's own output -- "
                              "{'layer_weight_bits': {...}, 'layer_act_bits': {...}}, one entry per real "
                              "LayerQuantENet quantizer SITE (NOT the coarser per-conv-layer joint_bits_"
                              "folding_ilp_perlayer.py output directly -- run that through expand_layer_bits.py "
                              "first).")
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-calibration-images", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.layer_bits_file) as f:
        layer_bits = json.load(f)
    layer_weight_bits = layer_bits["layer_weight_bits"]
    layer_act_bits = layer_bits["layer_act_bits"]

    source_model_folder = NNUNET_RESULTS / args.dataset_name / f"{args.source_net_name}__{args.plans_name}__{args.configuration}"
    source_checkpoint_path = source_model_folder / f"fold_{args.fold}" / args.source_checkpoint_name
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint_path}")

    print(f"Loading source FP32 checkpoint: {source_checkpoint_path}")
    quant_model = LayerQuantENet.from_pretrained(
        source_checkpoint_path, layer_weight_bits, layer_act_bits,
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, separable_dilated=True, trainable_slope=False,
    )

    print(f"Calibrating on real preprocessed images (device={args.device})...")
    calibration_batches = load_calibration_batches(args.dataset_name, args.n_calibration_images, seed=args.calibration_seed)
    n_used = calibrate(quant_model, calibration_batches, args.device, seed=args.calibration_seed)
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
    new_checkpoint["trainer_name"] = "nnUNetTrainerLayerQuantENet_27_2_reg_trailing_perlayer"
    out_checkpoint_path = out_fold_dir / "checkpoint_best.pth"
    torch.save(new_checkpoint, out_checkpoint_path)
    print(f"Saved calibrated checkpoint: {out_checkpoint_path}")


if __name__ == "__main__":
    main()
