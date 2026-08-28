"""Post-training quantization (PTQ) for the PER-BLOCK-quantized QuantENet
variants (QuantENet26_5_w24, QuantENetS19Block) -- same calibrate-only, no-
retraining pipeline as ptq.py (see its own module docstring for the full
PTQ-vs-QAT rationale, not repeated here), generalized to a per-block bit
assignment (compression/hawq/ilp_search.py's per-block output) instead of
one homogeneous weight_bit_width/act_bit_width pair, and to two hardcoded
architectures instead of ptq.py's fully-flag-driven QuantENet.

Reuses each model class's own from_pretrained (already does the exact
strict=False name+shape weight transfer ptq.py's own transfer_fp32_weights
implements by hand) instead of duplicating that logic -- this file only
adds calibration + nnU-Net-checkpoint-folder packaging on top.

Usage (26_5_w24, decomposed-nonneg-PReLU -- see QuantENet26_5_w24.py's own
module docstring for why this specific combination is a deliberate,
flagged choice, not a like-for-like nonneg_block substitute):
    python compression/post-quantization/ptq_block.py \\
        --model 26_5_w24 \\
        --source-net-name nnUNetTrainerENet_26_5_w24 \\
        --out-net-name nnUNetTrainerENetQuant_26_5_w24_ptq_block \\
        --block-bits-file compression/hawq/block_bits_26_5_w24.json \\
        --leaky-slope-map-file compression/post-quantization/slope_maps/26_5_w24.json

Usage (S19, real trained nonneg_block slopes):
    python compression/post-quantization/ptq_block.py \\
        --model s19 \\
        --source-net-name nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid \\
        --out-net-name nnUNetTrainerENetQuantS19_ptq_block \\
        --block-bits-file compression/hawq/block_bits_s19.json \\
        --leaky-slope-map-file compression/post-quantization/slope_maps/19_reginterleaved_separable_nonneg_block_double_mid.json \\
        --source-checkpoint-name checkpoint_final.pth

--prune-blocks (optional, comma-separated dotted block names, e.g.
"stage3.1,regular4.1"): applied via ENet.py's own apply_block_pruning
AFTER from_pretrained's weight transfer but BEFORE calibration -- pruned
blocks become nn.Identity() (pure skip, zero compute), and calibration
then runs on the ALREADY-PRUNED graph so every surviving quantizer's scale
reflects the real post-pruning activation distribution, not the original
network's. Caller's responsibility to only name channel-preserving
residual blocks (RegularBottleneck-style) -- never down1/down2/up4/up5/
initial/final, which change channel counts (see apply_block_pruning's own
docstring in ENet.py).

Usage (26_9_w24_s14w12_nonneg_block, vanilla INT8 vs. pruned-INT8
comparison -- real trained nonneg_block slopes):
    python compression/post-quantization/ptq_block.py \\
        --model 26_9_w24_s14w12_nonneg_block \\
        --source-net-name nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block \\
        --out-net-name nnUNetTrainerENetQuant_26_9_ptq_int8 \\
        --block-bits-file compression/hawq/block_bits_26_9_w24_s14w12_nonneg_block_int8.json \\
        --leaky-slope-map-file compression/post-quantization/slope_maps/26_9_w24_s14w12_nonneg_block.json
    python compression/post-quantization/ptq_block.py \\
        --model 26_9_w24_s14w12_nonneg_block \\
        --source-net-name nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block \\
        --out-net-name nnUNetTrainerENetQuant_26_9_ptq_int8_pruned5 \\
        --block-bits-file compression/hawq/block_bits_26_9_w24_s14w12_nonneg_block_int8.json \\
        --leaky-slope-map-file compression/post-quantization/slope_maps/26_9_w24_s14w12_nonneg_block.json \\
        --prune-blocks stage3.1,regular4.1,stage3.6,stage3.3,stage3.7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import apply_block_pruning  # noqa: E402
from nnunetv2.nets.QuantENet26_5_w24 import BLOCK_NAMES as BLOCK_NAMES_26_5_W24, QuantENet26_5_w24  # noqa: E402
from nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block import (  # noqa: E402
    BLOCK_NAMES as BLOCK_NAMES_26_9, QuantENet26_9_w24_s14w12_nonneg_block,
)
from nnunetv2.nets.QuantENetS19Block import BLOCK_NAMES as BLOCK_NAMES_S19, QuantENetS19Block  # noqa: E402

from brevitas.graph.calibrate import calibration_mode  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"

MODELS = {
    "26_5_w24": {
        "cls": QuantENet26_5_w24, "block_names": BLOCK_NAMES_26_5_W24,
        "trainer_name": "nnUNetTrainerENetQuant26_5_w24Block",
    },
    "s19": {
        "cls": QuantENetS19Block, "block_names": BLOCK_NAMES_S19,
        "trainer_name": "nnUNetTrainerENetQuantS19Block",
    },
    "26_9_w24_s14w12_nonneg_block": {
        "cls": QuantENet26_9_w24_s14w12_nonneg_block, "block_names": BLOCK_NAMES_26_9,
        "trainer_name": "nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock",
    },
}


def load_calibration_batches(dataset_name: str, n_images: int, seed: int = 0) -> list[torch.Tensor]:
    """Same real-preprocessed-image sampling as ptq.py's own
    load_calibration_batches (batch_size fixed at 1 -- real cases have
    varying H/W, see that function's own docstring for why)."""
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not image_files:
        raise FileNotFoundError(f"No preprocessed .npy images found under {preprocessed_dir}")
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_images, len(image_files)))
    return [torch.from_numpy(np.load(p)).float() for p in sampled]


def calibrate(quant_model: torch.nn.Module, calibration_batches: list[torch.Tensor], device: str) -> int:
    """Identical to ptq.py's own calibrate() -- see its docstring."""
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


def save_calibrated_checkpoint(
    quant_model: torch.nn.Module, reference_model_folder: Path, reference_checkpoint_name: str,
    out_net_name: str, dataset_name: str, plans_name: str, configuration: str, fold: int, trainer_name: str,
) -> Path:
    """Identical packaging to ptq.py's own save_calibrated_checkpoint (see
    its docstring for the state_dict()+named_parameters(remove_duplicate=
    False) merge rationale) -- just stamps the caller-given per-block
    trainer_name instead of the fixed "nnUNetTrainerENetQuant"."""
    reference_checkpoint_path = reference_model_folder / f"fold_{fold}" / reference_checkpoint_name
    reference_checkpoint = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)

    out_model_folder = NNUNET_RESULTS / dataset_name / f"{out_net_name}__{plans_name}__{configuration}"
    out_fold_dir = out_model_folder / f"fold_{fold}"
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    for meta_file in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        src = reference_model_folder / meta_file
        if src.exists():
            (out_model_folder / meta_file).write_bytes(src.read_bytes())

    network_weights = dict(quant_model.state_dict())
    network_weights.update(dict(quant_model.named_parameters(remove_duplicate=False)))
    new_checkpoint = dict(reference_checkpoint)
    new_checkpoint["network_weights"] = network_weights
    new_checkpoint["trainer_name"] = trainer_name
    out_checkpoint_path = out_fold_dir / "checkpoint_best.pth"
    torch.save(new_checkpoint, out_checkpoint_path)
    return out_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--source-net-name", required=True, help="config_name of the already-trained FP32 ENet checkpoint to quantize.")
    parser.add_argument("--out-net-name", required=True, help="net-name for the new calibrated PTQ checkpoint -- pass this to collect_results.py --net-name afterward.")
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--block-bits-file", required=True, type=Path, help="compression/hawq/ilp_search.py's per-block output JSON.")
    parser.add_argument("--leaky-slope-map-file", default=None, type=Path)
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--n-calibration-images", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prune-blocks", default=None,
        help="Comma-separated dotted block names (e.g. 'stage3.1,regular4.1') to replace with nn.Identity() "
             "via ENet.py's apply_block_pruning, applied AFTER from_pretrained but BEFORE calibration -- only "
             "channel-preserving residual blocks are safe (never down1/down2/up4/up5/initial/final).",
    )
    args = parser.parse_args()

    spec = MODELS[args.model]
    with open(args.block_bits_file) as f:
        block_bits = json.load(f)
    missing = [b for b in spec["block_names"] if b not in block_bits["stage_weight_bits"] or b not in block_bits["stage_act_bits"]]
    if missing:
        raise ValueError(f"{args.block_bits_file} is missing entries for blocks: {missing}")

    leaky_slope_map = None
    if args.leaky_slope_map_file:
        with open(args.leaky_slope_map_file) as f:
            leaky_slope_map = json.load(f)

    source_model_folder = NNUNET_RESULTS / args.dataset_name / f"{args.source_net_name}__{args.plans_name}__{args.configuration}"
    source_checkpoint_path = source_model_folder / f"fold_{args.fold}" / args.source_checkpoint_name
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Source FP32 checkpoint not found: {source_checkpoint_path}")

    quant_model = spec["cls"].from_pretrained(
        source_checkpoint_path, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
    )

    if args.prune_blocks:
        prune_names = [name.strip() for name in args.prune_blocks.split(",") if name.strip()]
        n_pruned = apply_block_pruning(quant_model, prune_names)
        if n_pruned != len(prune_names):
            raise ValueError(f"--prune-blocks {args.prune_blocks!r} -- expected {len(prune_names)} blocks pruned, got {n_pruned}.")
        print(f"Pruned {n_pruned} block(s) to nn.Identity() before calibration: {prune_names}")

    calibration_batches = load_calibration_batches(args.dataset_name, args.n_calibration_images, seed=args.calibration_seed)
    print(f"Calibrating on up to {len(calibration_batches)} real preprocessed images...")
    n_used = calibrate(quant_model, calibration_batches, args.device)
    print(f"Calibration used {n_used}/{len(calibration_batches)} images (some may have been skipped for shape reasons).")

    out_checkpoint_path = save_calibrated_checkpoint(
        quant_model, source_model_folder, args.source_checkpoint_name, args.out_net_name,
        args.dataset_name, args.plans_name, args.configuration, args.fold, spec["trainer_name"],
    )
    print(f"Saved calibrated PTQ checkpoint: {out_checkpoint_path} (trainer_name={spec['trainer_name']})")


if __name__ == "__main__":
    main()
