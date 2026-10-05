"""Export the PTQ-calibrated S12-dense 256x256 network (U4 widths (4,16,32,16,4),
bottlenecks (4,8,8,2,1), dense_dilation context, plain decoder_type="nearest_upsample",
Dataset510_ARCADE_256_4c, uniform INT6) to a FINN-safe LayerQuantEnetFINN ONNX, for the
MILP/artifacts/S12_dense_256_u4_analytical_v1/int6_fps250_lat200/ folding/FIFO build (see
that artifact's README.md).

Checkpoint is POST-TRAINING quantization only (no fine-tuning): built by
compression/post-quantization/calibrate_12_dense_relu_nearest_upsample_256_perlayer.py, which
loads the real FP32 nearest_upsample weights (trained at 512x512, resolution-independent) and
runs a single Brevitas calibration_mode() pass over all 1200 Dataset510_ARCADE_256_4c images
(reflect-padded to a multiple of 8) to set runtime-stats activation scales -- no backprop, no
new weights. See that script's docstring and this checkpoint's sibling calibration_report.json
for the exact provenance. --calibrate here (off by default) would run an ADDITIONAL calibration
pass on top of that; only useful as a sanity check, not required for a normal export.

checkpoint_best.pth arrives via the repo's git-ignore-safe transfer convention as
checkpoint_best.pth.txt (byte-identical copy); rename it to checkpoint_best.pth before running
this script if that hasn't been done yet (see /memories/repo/s12_dense_256_u4_analytical_v1_build.md).

Usage (inside the pytorch training container, repo root):
    docker exec <container> python3 hardware/builds/S12_dense_256_u4_analytical_v1/finn_export_S12_dense_256_u4_analytical_v1_finn_calibrated.py --skip-cross-test

Output: hardware/builds/S12_dense_256_u4_analytical_v1/outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated.onnx
Then, inside the FINN container:
    docker cp hardware/builds/S12_dense_256_u4_analytical_v1/outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# shared cross-test/calibration helpers (dataset-level, decoder-agnostic -- see that module).
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))

import nnunetv2.nets.CombinedQuantENet as _cq  # noqa: E402
import nnunetv2.nets.ENet as _en  # noqa: E402
import nnunetv2.nets.LayerQuantENet as _lq  # noqa: E402
import nnunetv2.nets.LayerQuantEnetFINN as _fin  # noqa: E402
import nnunetv2.nets.QuantENet as _qe  # noqa: E402


def _patch_missing_names() -> None:
    """LayerQuantENet.py in the working tree lost its `from QuantENet / ENet / CombinedQuantENet
    import (...)` block, VALID_CONTEXT_PATTERNS and ACT_SITE_TYPES in commit 66e0590e8b (intact at
    b16b35468a), and LayerQuantEnetFINN.py uses the same names through it. Inject what is missing
    (no-op when the files are intact) instead of editing them -- same patch as
    compression/post-quantization/calibrate_12_dense_relu_nearest_upsample_256_perlayer.py."""
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


_patch_missing_names()

import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
from finn_export_12_dense_relu_nearest_conv_upsample_256_trained import (  # noqa: E402
    DATASET_NAME,
    load_calibration_images,
    calibrate_runtime_stats,
    run_finn_inference,
    evaluate,
    compare_to_real_predictions,
)

# Redirect export_model()'s output into this job's own folder instead of the
# shared (deprecated) hardware/outputs/finn_exports/.
finn_enet_prod_export.OUT_DIR = Path(__file__).resolve().parent / "outputs"

CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DECODER_TYPE = "nearest_upsample"

NET_NAME = "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_256_uniform_int6_calibrated_all1200pad8"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / DATASET_NAME
    / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_256_u4_analytical_v1" / "int6_fps250_lat200"
    / "layer_bits_SITES_final.json"
)
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "nnUNetPlans_2d"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw" / DATASET_NAME
DEFAULT_REAL_PRED_DIR = NNUNET_RAW / f"labelsPr_{NET_NAME}"


def load_layer_bits(bits_file: Path, weight_names, act_names):
    """layer_bits_SITES_*.json (per-quantizer-site bits, {'layer_weight_bits': {...}, 'layer_act_bits': {...}})
    -- NOT the sibling layer_bits_folding_*.json. No fallback: every site this architecture needs must be present."""
    with open(bits_file) as f:
        raw = json.load(f)
    raw_w, raw_a = raw["layer_weight_bits"], raw["layer_act_bits"]
    missing_w = [n for n in weight_names if n not in raw_w]
    missing_a = [n for n in act_names if n not in raw_a]
    if missing_w or missing_a:
        raise KeyError(f"{bits_file}: missing weight sites {missing_w}, act sites {missing_a}")
    return {n: raw_w[n] for n in weight_names}, {n: raw_a[n] for n in act_names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--real-pred-dir", default=str(DEFAULT_REAL_PRED_DIR))
    parser.add_argument("--calibration-images", type=int, default=200,
                         help="only used if --calibrate is passed; -1 uses the full train split")
    parser.add_argument("--calibrate", action="store_true",
                         help="checkpoint is already PTQ-calibrated (Brevitas calibration_mode over all "
                              "1200 images) -- skipped by default; pass this to force an EXTRA runtime-stats pass")
    parser.add_argument("--skip-cross-test", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path} (did you rename checkpoint_best.pth.txt -> "
                     "checkpoint_best.pth? *.pth is git-ignored, so it transfers with a .txt suffix)")

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building REAL-weight, PTQ-calibrated, per-layer-bit-width FINN-safe "
          f"S12_dense_256_u4_analytical_v1 -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        checkpoint_path, layer_weight_bits, layer_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE,
    ).eval()

    if args.calibrate:
        print("\n=== Calibrating runtime-stats activation scales on real training data (EXTRA pass) ===")
        n_calib = None if args.calibration_images < 0 else args.calibration_images
        calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
        calibrate_runtime_stats(model, calibration_images)
    else:
        print("\n=== Skipping calibration -- checkpoint already PTQ-calibrated by "
              "compression/post-quantization/calibrate_12_dense_relu_nearest_upsample_256_perlayer.py ===")

    if not args.skip_export:
        print("\n=== Forward-pass sanity check + QONNX export ===")
        dummy = torch.rand(1, args.in_channels, 256, 256) * 2 - 1
        with torch.no_grad():
            out = model(dummy)
        assert out.shape[2:] == (256, 256), f"output HxW {tuple(out.shape[2:])} != (256,256)"
        assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
        print(f"  forward OK: output shape {tuple(out.shape)}")
        name = "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated"
        export_model(model, name, dummy)
        print("\nExported. Copy to FINN container with:")
        print(f"  docker cp hardware/builds/S12_dense_256_u4_analytical_v1/outputs/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")

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
