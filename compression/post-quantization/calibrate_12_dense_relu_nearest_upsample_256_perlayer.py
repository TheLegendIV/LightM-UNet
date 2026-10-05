"""Builds the S12 nearest-decoder network (nnUNetTrainerENet_12_dense_relu_nearest_upsample, U4 widths, dense_dilation, plain nearest decoder, 24,261 FP32
parameters) as a `LayerQuantEnetFINN` at the per-site bit widths of a layer-bits file (uniform INT6 by default), CALIBRATES the new Brevitas quantizers on the
256x256 images (Dataset510_ARCADE_256_4c) and saves the result as a transferable checkpoint. Post-training quantization only: no fine-tuning.

Why a new script: the existing 256x256 sibling (calibrate_12_dense_relu_nearest_conv_upsample_256_perlayer.py) is the conv-decoder / dense_dilation_half network and
loads its FP32 weights from the same dataset. The nearest-decoder S12 net was trained only at 512x512 (Dataset509_ARCADE_1x1_4c); its weights are
resolution-independent (fully convolutional), so they are loaded from there and calibrated on the 256x256 data. Dataset510 has 1200 images (1000 train + 200 val);
the default calibrates on ALL of them (train + val), reflect-padded to a multiple of 8, as the other 'all1200pad8' S12 calibrations do. The calibrated model is
therefore not held-out on the validation images.

Output (under data/nnUNet_results/<dataset>/<out-net-name>__nnUNetPlans__2d/fold_0/):
  checkpoint_best.pth      nnU-Net checkpoint (network_weights = the calibrated LayerQuantEnetFINN state dict; trainer_name = the FINN nearest_upsample perlayer trainer)
  checkpoint_best.pth.txt  byte-identical copy (the .txt name is the transfer convention: *.pth is git-ignored)
  layer_bits_SITES.json    the per-site bits the network was built with
  calibration_report.json  source checkpoint, images used, clamped scales
plus dataset.json / plans.json / dataset_fingerprint.json of the 256x256 dataset next to it (so nnUNetv2_predict finds the 256 patch size).

Usage (in lightmunet_dev, repo root):
    python compression/post-quantization/calibrate_12_dense_relu_nearest_upsample_256_perlayer.py \\
        --layer-bits-file MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/layer_bits_SITES_final.json \\
        --out-net-name nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_256_uniform_int6_calibrated_all1200pad8
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
import nnunetv2.nets.CombinedQuantENet as _cq  # noqa: E402
import nnunetv2.nets.ENet as _en  # noqa: E402
import nnunetv2.nets.LayerQuantENet as _lq  # noqa: E402
import nnunetv2.nets.LayerQuantEnetFINN as _fin  # noqa: E402
import nnunetv2.nets.QuantENet as _qe  # noqa: E402

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
NNUNET_RESULTS = REPO_ROOT / "data" / "nnUNet_results"

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DECODER_TYPE = "nearest_upsample"
TRAINER_NAME = "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer"


def patch_missing_names() -> None:
    """LayerQuantENet.py in the working tree lost its `from QuantENet / ENet / CombinedQuantENet import (...)` block, VALID_CONTEXT_PATTERNS and ACT_SITE_TYPES in
    commit 66e0590e8b (intact at b16b35468a), and LayerQuantEnetFINN.py uses the same names through it. Inject what is missing (no-op when the files are intact)
    instead of editing them."""
    import brevitas.nn as qnn
    for mod in (_qe, _cq, _en):
        for name in dir(mod):
            if not name.startswith("__") and not hasattr(_lq, name):
                setattr(_lq, name, getattr(mod, name))
    if not hasattr(_lq, "ACT_SITE_TYPES"):
        _lq.ACT_SITE_TYPES = (qnn.QuantReLU, qnn.QuantIdentity, _qe.QuantDecomposedLeakyAct, _qe.QuantFusedLeakyAct, qnn.QuantEltwiseAdd)
    for mod in (_lq, _qe, _cq, _en):
        for name in dir(mod):
            if not name.startswith("__") and not hasattr(_fin, name):
                setattr(_fin, name, getattr(mod, name))


patch_missing_names()
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def load_calibration_batches(dataset_name: str, n_images: int, seed: int = 0, pad_multiple: int = 0) -> list[torch.Tensor]:
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    if not image_files:
        raise FileNotFoundError(f"No preprocessed .npy images found under {preprocessed_dir}")
    if n_images > 0:
        image_files = random.Random(seed).sample(image_files, k=min(n_images, len(image_files)))
    tensors = [torch.from_numpy(np.load(p)).float() for p in image_files]
    return [_pad_to_multiple(t, pad_multiple) for t in tensors] if pad_multiple else tensors


# fp16 min-normal is 6.1e-5: a calibrated activation scale below it underflows to 0 under CUDA fp16 autocast (nnUNetv2_predict) and turns every quantize-dequantize
# on that site into x/0 -> NaN (the regular5.0 incident of the sibling calibrate scripts).
FP16_SAFE_SCALE_FLOOR = 1e-4


def clamp_quantizer_scales(quant_model: torch.nn.Module, floor: float = FP16_SAFE_SCALE_FLOOR) -> int:
    n_clamped = 0
    with torch.no_grad():
        for name, param in quant_model.named_parameters():
            if name.endswith("scaling_impl.value"):
                too_small = param.data.abs() < floor
                if too_small.any():
                    print(f"  [clamp] {name}: {param.data[too_small].tolist()} -> {floor}")
                    param.data[too_small] = floor
                    n_clamped += int(too_small.sum().item())
    return n_clamped


def calibrate(quant_model: torch.nn.Module, calibration_batches: list[torch.Tensor], device: str, seed: int) -> int:
    """Brevitas calibration pass in eval mode (the FINN class has no dropout on this path, and the FINN export script calibrates in eval mode too)."""
    torch.manual_seed(seed)
    quant_model.to(device)
    quant_model.eval()
    n_used = 0
    with torch.no_grad(), calibration_mode(quant_model):
        for i, batch in enumerate(calibration_batches):
            try:
                quant_model(batch.to(device))
                n_used += 1
            except RuntimeError as error:
                print(f"  [skip] calibration image {i} with shape {tuple(batch.shape)} failed forward pass: {error}")
            if (i + 1) % 200 == 0:
                print(f"  calibrated {i + 1}/{len(calibration_batches)}", flush=True)
    quant_model.eval()
    if n_used == 0:
        raise RuntimeError("Every calibration image failed -- can't calibrate quantizer scales at all.")
    return n_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dataset-name", default="Dataset509_ARCADE_1x1_4c", help="where the FP32 nearest_upsample weights were trained (512x512)")
    parser.add_argument("--source-net-name", default="nnUNetTrainerENet_12_dense_relu_nearest_upsample_warmstart150ep")
    parser.add_argument("--source-checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--dataset-name", default="Dataset510_ARCADE_256_4c", help="calibration data and output location (256x256)")
    parser.add_argument("--meta-source-net-name", default="nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256",
                        help="a net trained on --dataset-name: its plans.json / dataset.json / checkpoint metadata (init_args) are copied so the 256 patch size is used")
    parser.add_argument("--out-net-name", required=True)
    parser.add_argument("--layer-bits-file", required=True, type=Path,
                        help="{'layer_weight_bits': {...}, 'layer_act_bits': {...}} per quantizer SITE (MILP/expand_layer_bits.py output)")
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-calibration-images", type=int, default=-1, help="-1 (default) = every preprocessed image (train + val)")
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--pad-to-multiple", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    layer_bits = json.loads(args.layer_bits_file.read_text())
    layer_weight_bits, layer_act_bits = layer_bits["layer_weight_bits"], layer_bits["layer_act_bits"]

    source_folder = NNUNET_RESULTS / args.source_dataset_name / f"{args.source_net_name}__{args.plans_name}__{args.configuration}"
    source_checkpoint_path = source_folder / f"fold_{args.fold}" / args.source_checkpoint_name
    meta_folder = NNUNET_RESULTS / args.dataset_name / f"{args.meta_source_net_name}__{args.plans_name}__{args.configuration}"
    meta_checkpoint_path = meta_folder / f"fold_{args.fold}" / "checkpoint_best.pth"
    for p in (source_checkpoint_path, meta_checkpoint_path):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"FP32 source: {source_checkpoint_path}")
    quant_model = LayerQuantEnetFINN.from_pretrained(
        source_checkpoint_path, layer_weight_bits, layer_act_bits, in_channels=1, out_channels=5, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN, decoder_type=DECODER_TYPE,
    )

    print(f"Calibrating on {args.dataset_name} (device={args.device}) ...")
    batches = load_calibration_batches(args.dataset_name, args.n_calibration_images, seed=args.calibration_seed, pad_multiple=args.pad_to_multiple)
    n_used = calibrate(quant_model, batches, args.device, seed=args.calibration_seed)
    print(f"Calibration used {n_used}/{len(batches)} images.")
    quant_model.to("cpu")
    n_clamped = clamp_quantizer_scales(quant_model)
    print(f"Clamped {n_clamped} quantizer scale(s) below the fp16-safe floor ({FP16_SAFE_SCALE_FLOOR:.0e}).")

    out_model_folder = NNUNET_RESULTS / args.dataset_name / f"{args.out_net_name}__{args.plans_name}__{args.configuration}"
    out_fold_dir = out_model_folder / f"fold_{args.fold}"
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    for meta_file in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        src = meta_folder / meta_file
        if src.exists():
            (out_model_folder / meta_file).write_bytes(src.read_bytes())

    checkpoint = dict(torch.load(meta_checkpoint_path, map_location="cpu", weights_only=False))
    network_weights = dict(quant_model.state_dict())
    network_weights.update(dict(quant_model.named_parameters(remove_duplicate=False)))
    checkpoint["network_weights"] = network_weights
    checkpoint["trainer_name"] = TRAINER_NAME
    for stale in ("optimizer_state", "grad_scaler_state"):      # belong to the metadata net, not to these weights
        checkpoint.pop(stale, None)
    out_pth = out_fold_dir / "checkpoint_best.pth"
    out_txt = out_fold_dir / "checkpoint_best.pth.txt"
    torch.save(checkpoint, out_pth)
    out_txt.write_bytes(out_pth.read_bytes())
    (out_fold_dir / "layer_bits_SITES.json").write_text(json.dumps(layer_bits, indent=2))
    (out_fold_dir / "calibration_report.json").write_text(json.dumps(dict(
        source_checkpoint=str(source_checkpoint_path), metadata_checkpoint=str(meta_checkpoint_path), layer_bits_file=str(args.layer_bits_file),
        calibration_dataset=args.dataset_name, images_found=len(batches), images_used=n_used, pad_to_multiple=args.pad_to_multiple,
        clamped_scales=n_clamped, model="LayerQuantEnetFINN", decoder_type=DECODER_TYPE, context_pattern=CONTEXT_PATTERN,
        weight_bits=sorted(set(layer_weight_bits.values())), act_bits=sorted(set(layer_act_bits.values()))), indent=2))
    print(f"Saved: {out_pth}\n       {out_txt}")


if __name__ == "__main__":
    main()
