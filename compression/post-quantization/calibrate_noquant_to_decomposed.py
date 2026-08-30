"""Reconstruct a real, exportable QuantDecomposedLeakyAct-based checkpoint
from one trained with quant_enabled=False (the fake-quant chain bypassed
entirely during training, pure float leaky computation -- see QuantENet.py's
own docstring) -- then CALIBRATE the newly-introduced pre_quant/act_pos/
out_quant quantizers on real data before saving.

Why calibration is required, not optional: quant_enabled=False's forward
pass never calls pre_quant/act_pos/out_quant at all, so their own scale
parameters sit at Brevitas's arbitrary default/uninitialized value the
entire training run -- confirmed by direct experiment (compression session
2026-08-29/30): building the reconstructed model and running inference with
NO calibration produced 100% empty predictions on every single test case
(a misleadingly nonzero mean per-class dice from the "class genuinely
absent, predicted absent" convention masked this until the raw prediction
files were inspected directly -- dice_binary/n_components being exactly 0.0
was the real tell). Calibration (Brevitas's calibration_mode(), a handful of
real forward passes, NO gradients/backprop/optimizer -- identical mechanism
to ptq_block.py's own calibrate()) fixes that by setting each quantizer's
scale from the real observed activation range instead of an arbitrary
default.

alpha and every conv/BN weight transfer UNCHANGED (real name+shape match,
same generic mechanism QuantENet26_9_w24_s14w12_nonneg_block.from_pretrained
already implements) -- calibration only touches pre_quant/act_pos/out_quant's
own scale, nothing else. Calibration alone was NOT found sufficient on its
own (real held-out eval after calibration: dice~0.028, dice_binary=0.141,
~996 predicted connected components/image -- real signal, but extremely
fragmented, since the network's weights were never exposed to ANY
quantization noise at these sites and calibration doesn't teach robustness
to the resulting rounding, only sets a sane scale) -- this checkpoint is
meant to be a WARM START for a short real fine-tuning continuation
(quant_enabled=True, gradients flowing normally) via nnUNetv2_train, not a
final deployable artifact on its own. See
compression/slurm/qat_26_9_w24_s14w12_nonneg_block_noquant_finetune_internal8.job
for that continuation.

Usage:
    python compression/post-quantization/calibrate_noquant_to_decomposed.py \\
        --source-net-name nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock_acc1x_joint_noquant_damped_zeroslope_5ep \\
        --out-net-name nnUNetTrainerENetQuant_26_9_noquant_zeroslope_internal8_calibrated \\
        --block-bits-file compression/hawq/block_bits_26_9_w24_s14w12_nonneg_block_acc1x_joint.json \\
        --leaky-slope-map-file compression/post-quantization/slope_maps/26_9_w24_s14w12_nonneg_block_zeroed.json \\
        --internal-bit-width 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block import QuantENet26_9_w24_s14w12_nonneg_block  # noqa: E402
from ptq_block import load_calibration_batches, calibrate  # noqa: E402

NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-net-name", required=True, help="config_name of the quant_enabled=False-trained checkpoint.")
    parser.add_argument("--source-checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--out-net-name", required=True)
    parser.add_argument("--dataset-name", default="Dataset509_ARCADE_1x1_4c")
    parser.add_argument("--block-bits-file", required=True, type=Path)
    parser.add_argument("--leaky-slope-map-file", required=True, type=Path)
    parser.add_argument("--internal-bit-width", type=int, default=8)
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-calibration-images", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.block_bits_file) as f:
        block_bits = json.load(f)
    with open(args.leaky_slope_map_file) as f:
        leaky_slope_map = json.load(f)

    source_model_folder = NNUNET_RESULTS / args.dataset_name / f"{args.source_net_name}__{args.plans_name}__{args.configuration}"
    source_checkpoint_path = source_model_folder / f"fold_{args.fold}" / args.source_checkpoint_name
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint_path}")

    print(f"Loading source (quant_enabled=False trained) checkpoint: {source_checkpoint_path}")
    quant_model = QuantENet26_9_w24_s14w12_nonneg_block.from_pretrained(
        source_checkpoint_path, block_bits["stage_weight_bits"], block_bits["stage_act_bits"], leaky_slope_map,
        trainable_slope=True, internal_bit_width=args.internal_bit_width,
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
    new_checkpoint["trainer_name"] = "nnUNetTrainerENetQuant26_9_w24_s14w12_nonneg_blockBlock"
    out_checkpoint_path = out_fold_dir / "checkpoint_best.pth"
    torch.save(new_checkpoint, out_checkpoint_path)
    print(f"Saved calibrated checkpoint: {out_checkpoint_path}")


if __name__ == "__main__":
    main()
