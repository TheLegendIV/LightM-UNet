"""v3 sibling of finn_export_12_dense_relu_nearest_conv_upsample_256_v3_trained.py: exports the S12 dense 256x256 full-width
network for the TIED-residual-bits MILP solve (MILP/artifacts/S12_dense_256_fullwidth_joinsdist_v1/final_tied_dsr1.04/:
--joins-distributed --tie-residual-bits, FPS 100, DSR 1.04, caps LUT 0.7 / BRAM 0.4 / DSP 0.9). Deltas vs v2:
  - Source is the FP32 checkpoint (nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256, trained from scratch, 150 ep),
    re-quantized with the tied SITES bits via LayerQuantEnetFINN.from_pretrained, then runtime-stats calibrated on the FULL
    train split (reflect-padded to x8) -- so --calibrate is ON by default here (v2 started from a FINN-native QAT checkpoint).
  - The add's input AND output quantizers use act_bits["residual_add"] (the network no longer hard-wires Int8).
  - Output goes to this folder's outputs/ with a "_v3_" tag.

Usage (pytorch container, repo root):
    docker exec lightmunet_dev python3 hardware/builds/12_dense_relu_nearest_conv_upsample_256_v3/finn_export_12_dense_relu_nearest_conv_upsample_256_v3_trained.py --skip-cross-test

Output: hardware/builds/12_dense_relu_nearest_conv_upsample_256_v3/outputs/quantEnet_12_dense_relu_nearest_conv_upsample_256_v3_trained.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "501_ARCADE"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# v1's job folder: only for the shared helpers below, not yet promoted to
# top-level hardware/ shared infra (see builds/README.md).
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))
import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
from finn_export_12_dense_relu_nearest_conv_upsample_256_trained import (  # noqa: E402
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    DECODER_TYPE,
    DATASET_NAME,
    load_layer_bits,
    load_calibration_images,
    calibrate_runtime_stats,
    run_finn_inference,
    evaluate,
    compare_to_real_predictions,
)

# Redirect export_model()'s output into this job's own folder instead of the
# shared (deprecated) hardware/outputs/finn_exports/.
finn_enet_prod_export.OUT_DIR = Path(__file__).resolve().parent / "outputs"

NET_NAME = "nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample_256"  # FP32 source (also the cross-test reference name)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / DATASET_NAME
    / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_256_fullwidth_joinsdist_v1" / "final_tied_dsr1.04"
    / "layer_bits_SITES_final.json"
)
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "nnUNetPlans_2d"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw" / DATASET_NAME
DEFAULT_REAL_PRED_DIR = NNUNET_RAW / f"labelsPr_{NET_NAME}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--real-pred-dir", default=str(DEFAULT_REAL_PRED_DIR))
    parser.add_argument("--calibration-images", type=int, default=-1,
                         help="only used if --calibrate is passed; -1 uses the full train split")
    parser.add_argument("--skip-calibration", action="store_true",
                         help="skip the runtime-stats calibration pass (default: calibrate on the full train split)")
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

    print(f"\n=== Building REAL-weight (v3, tied residual bits), per-layer-bit-width FINN-safe "
          f"12_dense_relu_nearest_conv_upsample_256 -- {len(weight_names)} weight sites, "
          f"{len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        checkpoint_path, layer_weight_bits, layer_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE,
    ).eval()

    if not args.skip_calibration:
        print("\n=== Calibrating runtime-stats activation scales on real training data ===")
        n_calib = None if args.calibration_images < 0 else args.calibration_images
        calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
        calibrate_runtime_stats(model, calibration_images)
    else:
        print("\n=== Skipping calibration -- checkpoint was fine-tuned directly on LayerQuantEnetFINN, "
              "runtime-stats scales already came in via from_pretrained's state-dict transfer ===")

    if not args.skip_export:
        print("\n=== Forward-pass sanity check + QONNX export ===")
        dummy = torch.rand(1, args.in_channels, 256, 256) * 2 - 1
        with torch.no_grad():
            out = model(dummy)
        assert out.shape[2:] == (256, 256), f"output HxW {tuple(out.shape[2:])} != (256,256)"
        assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
        print(f"  forward OK: output shape {tuple(out.shape)}")
        name = "quantEnet_12_dense_relu_nearest_conv_upsample_256_v3_trained"
        export_model(model, name, dummy)
        print(f"\nExported. Copy to FINN container with:")
        print(f"  docker cp hardware/builds/12_dense_relu_nearest_conv_upsample_256_v3/outputs/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")

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
