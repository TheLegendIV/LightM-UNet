"""Export the REAL, QAT fine-tuned nnUNetTrainerLayerQuantENet_12_dense_relu_
nearest_conv_upsample_256_perlayer checkpoint (256x256, decoder_type=
"nearest_conv_upsample", context_pattern="dense_dilation_half", Dataset510_
ARCADE_256_4c, dice=0.7819 per compression/results.csv) to a FINN-safe
LayerQuantEnetFINN (see enet/nnunetv2/nets/LayerQuantEnetFINN.py) via its own
from_pretrained, then QONNX-export it.

CROSS TEST: this decoder_type is the "closer" case in LayerQuantEnetFINN's
docstring (point 2b) -- up4/up5.main_up is a bare parameter-free
nn.Upsample(nearest) (not a frozen bilinear-kernel substitute) and
skip_resize_conv is the exact real trained op, so the ONLY structural
departure from the real LayerQuantENet is `initial`'s new branch_quant
(module docstring point 0). To confirm that substitution doesn't move
accuracy, this script also runs both the FINN mirror and (via its already-
computed official predictions) the real LayerQuantENet over the full 300-case
imagesTs/labelsTs test set and reports: (a) the FINN mirror's own dice
(same dice_score primitive, same preprocessing nnU-Net itself uses for this
dataset -- see below), (b) per-case prediction agreement between the FINN
mirror and the real model's official predictions.

Preprocessing note: replicated by hand here (z-score per-image, no
resampling, no tiling) rather than going through nnUNetPredictor, because
this dataset's plans (nnUNetPlans.json, config "2d") make that exact and
trivial: patch_size==[256,256] matches every imagesTs case exactly (no
sliding-window tiling), spacing==[1,1] (no resampling), normalization_schemes
==["ZScoreNormalization"] with use_mask_for_norm=False (per-image mean/std,
see enet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py).
collect_results.py's own run_inference also passes --disable_tta, so there's
no mirroring augmentation to replicate either -- a single forward pass here
is the exact same computation nnUNetv2_predict_from_modelfolder performs for
this dataset, not an approximation of it.

Usage (inside the pytorch training container):
    docker exec lightm_pytorch python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_nearest_conv_upsample_256_trained.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_nearest_conv_upsample_256_trained_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from brevitas.graph.calibrate import calibration_mode
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
import segmentation_topology as topo  # noqa: E402

DATASET_NAME = "Dataset510_ARCADE_256_4c"
CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation_half"
DECODER_TYPE = "nearest_conv_upsample"
FALLBACK_BITS = 6  # matches the convention in every other finn_export_*_dummy.py

NET_NAME = (
    "nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_256_perlayer_"
    "12_dense_relu_nearest_conv_upsample_256_joint_alpha1.0_perlayer_candidatebits468_"
    "forcedsp_lut50_bram50_dsp90_fps250_ft15ep"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / DATASET_NAME
    / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_nn_upsample_256_v1"
    / "layer_bits_SITES_12_dense_relu_nearest_conv_upsample_256_joint_alpha1.0_"
      "candidatebits468_forcedsp_lut50_bram50_dsp90_fps250.json"
)
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "nnUNetPlans_2d"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw" / DATASET_NAME
DEFAULT_REAL_PRED_DIR = NNUNET_RAW / f"labelsPr_{NET_NAME}"


def load_layer_bits(
    bits_file: Path, weight_names: tuple[str, ...], act_names: tuple[str, ...], fallback: int = FALLBACK_BITS,
) -> tuple[dict[str, int], dict[str, int]]:
    with open(bits_file) as f:
        raw = json.load(f)
    raw_w, raw_a = raw["layer_weight_bits"], raw["layer_act_bits"]
    missing_w = [n for n in weight_names if n not in raw_w]
    missing_a = [n for n in act_names if n not in raw_a]
    if missing_w or missing_a:
        print(f"  WARNING: {len(missing_w)} weight site(s) / {len(missing_a)} act site(s) missing from "
              f"{bits_file.name}, using fallback={fallback}: {missing_w + missing_a}")
    return ({n: raw_w.get(n, fallback) for n in weight_names},
            {n: raw_a.get(n, fallback) for n in act_names})


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """nnU-Net preprocessed patches are stored at native per-case size, not
    pre-cropped to plans patch_size -- confirmed even for this dataset
    (train_*_p0000.npy shapes include e.g. (244,200), (255,256), not just
    (256,256)) -- so this padding is still needed despite imagesTs itself
    being uniformly 256x256 (a separate, already-resized raw split)."""
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def load_calibration_images(preprocessed_dir: Path, n: int | None, split: str = "train") -> list[torch.Tensor]:
    paths = sorted(preprocessed_dir.glob(f"{split}_*_p0000.npy"))
    if not paths:
        raise FileNotFoundError(f"no {split}_*_p0000.npy patches found under {preprocessed_dir}")
    if n is not None:
        paths = paths[:n]
    images = [_pad_to_multiple(torch.from_numpy(np.load(p)).float()) for p in paths]
    print(f"  Loaded {len(images)} real calibration images from {preprocessed_dir} (split={split}).")
    return images


def calibrate_runtime_stats(model: "LayerQuantEnetFINN", calibration_images: list[torch.Tensor]) -> None:
    with torch.no_grad(), calibration_mode(model):
        for image in calibration_images:
            model(image)
    print(f"  Calibrated runtime-stats scaling buffers over {len(calibration_images)} real training patches.")


def zscore(arr: np.ndarray) -> np.ndarray:
    """Byte-for-byte the same formula as enet/nnunetv2/preprocessing/normalization/
    default_normalization_schemes.py's ZScoreNormalization with use_mask_for_norm=False."""
    arr = arr.astype(np.float32)
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / max(std, 1e-8)


def run_finn_inference(model: "LayerQuantEnetFINN", images_ts_dir: Path) -> dict[str, np.ndarray]:
    """case_id -> (H,W) uint8 predicted class-id map, one forward pass per
    case (no tiling/mirroring needed, see module docstring)."""
    predictions: dict[str, np.ndarray] = {}
    paths = sorted(images_ts_dir.glob("*_0000.png"))
    with torch.no_grad():
        for i, path in enumerate(paths):
            case_id = path.stem.rsplit("_", 1)[0]  # strip nnU-Net's "_0000" channel suffix
            arr = zscore(np.asarray(Image.open(path)))
            x = torch.from_numpy(arr).float()[None, None]
            logits = model(x)
            pred = logits.argmax(dim=1)[0].to(torch.uint8).numpy()
            predictions[case_id] = pred
            if (i + 1) % 50 == 0:
                print(f"  FINN inference: {i + 1}/{len(paths)} cases")
    return predictions


def read_dataset_labels(dataset_name: str) -> dict[str, int]:
    with open(NNUNET_RAW.parent / dataset_name / "dataset.json") as f:
        return json.load(f)["labels"]


def foreground_class_ids_and_names(labels: dict[str, int]) -> list[tuple[int, str]]:
    return sorted((class_id, name) for name, class_id in labels.items() if class_id != 0)


def evaluate(predictions: dict[str, np.ndarray], labels_ts_dir: Path) -> dict:
    """Same per-class + binary dice convention as compression/collect_results.py's
    compute_eval_metrics, applied directly to an in-memory prediction dict
    instead of a labelsPr_* directory of PNGs."""
    class_ids_names = foreground_class_ids_and_names(read_dataset_labels(DATASET_NAME))
    per_class_dice = {name: [] for _, name in class_ids_names}
    binary_dice_list = []
    for path in sorted(labels_ts_dir.glob("*.png")):
        case_id = path.stem
        if case_id not in predictions:
            continue
        gt = np.asarray(Image.open(path)).astype(np.uint8)
        pred = predictions[case_id]
        for class_id, name in class_ids_names:
            per_class_dice[name].append(topo.dice_score(gt == class_id, pred == class_id))
        binary_dice_list.append(topo.dice_score(gt > 0, pred > 0))
    result = {f"dice_{name}": sum(v) / len(v) for name, v in per_class_dice.items()}
    result["dice"] = sum(result[f"dice_{name}"] for _, name in class_ids_names) / len(class_ids_names)
    result["dice_binary"] = sum(binary_dice_list) / len(binary_dice_list)
    result["n_cases"] = len(binary_dice_list)
    return result


def compare_to_real_predictions(predictions: dict[str, np.ndarray], real_pred_dir: Path) -> dict:
    """Direct FINN-mirror-vs-real-model prediction agreement -- the most
    direct signal for whether the FINN-safe substitutions changed the
    model's actual output, independent of ground truth."""
    class_ids_names = foreground_class_ids_and_names(read_dataset_labels(DATASET_NAME))
    per_class_agree_dice = {name: [] for _, name in class_ids_names}
    pixel_agree, pixel_total = 0, 0
    n_cases = 0
    for path in sorted(real_pred_dir.glob("*.png")):
        case_id = path.stem
        if case_id not in predictions:
            continue
        real_pred = np.asarray(Image.open(path)).astype(np.uint8)
        finn_pred = predictions[case_id]
        n_cases += 1
        pixel_agree += int((real_pred == finn_pred).sum())
        pixel_total += real_pred.size
        for class_id, name in class_ids_names:
            per_class_agree_dice[name].append(topo.dice_score(real_pred == class_id, finn_pred == class_id))
    return {
        "n_cases": n_cases,
        "pixel_agreement": pixel_agree / pixel_total if pixel_total else float("nan"),
        **{f"agree_dice_{name}": sum(v) / len(v) for name, v in per_class_agree_dice.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--real-pred-dir", default=str(DEFAULT_REAL_PRED_DIR))
    parser.add_argument("--calibration-images", type=int, default=200,
                         help="number of real training patches to calibrate on; -1 uses the full train split")
    parser.add_argument("--skip-cross-test", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building REAL-weight, per-layer-bit-width FINN-safe 12_dense_relu_nearest_conv_upsample_256 "
          f"-- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        checkpoint_path, layer_weight_bits, layer_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE,
    ).eval()

    print("\n=== Calibrating runtime-stats activation scales on real training data ===")
    n_calib = None if args.calibration_images < 0 else args.calibration_images
    calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
    calibrate_runtime_stats(model, calibration_images)

    if not args.skip_export:
        print("\n=== Forward-pass sanity check + QONNX export ===")
        dummy = torch.rand(1, args.in_channels, 256, 256) * 2 - 1
        with torch.no_grad():
            out = model(dummy)
        assert out.shape[2:] == (256, 256), f"output HxW {tuple(out.shape[2:])} != (256,256)"
        assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
        print(f"  forward OK: output shape {tuple(out.shape)}")
        name = "quantEnet_12_dense_relu_nearest_conv_upsample_256_trained_int8"
        export_model(model, name, dummy)
        print(f"\nExported. Copy to FINN container with:")
        print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")

    if not args.skip_cross_test:
        print("\n=== Cross test: FINN mirror vs. real model, full imagesTs/labelsTs (300 cases) ===")
        images_ts_dir = NNUNET_RAW / "imagesTs"
        labels_ts_dir = NNUNET_RAW / "labelsTs"
        predictions = run_finn_inference(model, images_ts_dir)

        finn_metrics = evaluate(predictions, labels_ts_dir)
        print(f"\nFINN mirror dice vs. ground truth ({finn_metrics['n_cases']} cases):")
        for k, v in finn_metrics.items():
            if k != "n_cases":
                print(f"  {k}: {v:.4f}")
        print(f"\nReference (real LayerQuantENet, official pipeline, from compression/results.csv): dice=0.7819")

        real_pred_dir = Path(args.real_pred_dir)
        if real_pred_dir.exists():
            agreement = compare_to_real_predictions(predictions, real_pred_dir)
            print(f"\nFINN mirror vs. real model prediction agreement ({agreement['n_cases']} cases):")
            for k, v in agreement.items():
                if k != "n_cases":
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"\n  WARNING: real-pred-dir not found ({real_pred_dir}), skipping agreement comparison.")


if __name__ == "__main__":
    main()
