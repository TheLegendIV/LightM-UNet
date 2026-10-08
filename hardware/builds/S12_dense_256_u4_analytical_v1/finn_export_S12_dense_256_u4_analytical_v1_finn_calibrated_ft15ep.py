"""Export the FINE-TUNED (15 epochs, on top of the PTQ-calibrated checkpoint) S12-dense
256x256 network -- IDENTICAL architecture/bits to finn_export_S12_dense_256_u4_analytical_v1_
finn_calibrated.py (U4 widths (4,16,32,16,4), bottlenecks (4,8,8,2,1), dense_dilation context,
decoder_type="nearest_upsample", Dataset510_ARCADE_256_4c, uniform INT6, same MILP/artifacts/
S12_dense_256_u4_analytical_v1/int6_fps250_lat200/ folding/FIFO build) -- only NET_NAME/
DEFAULT_CHECKPOINT and the exported `name` differ, so conv_order.json and the MILP folding
json are reused unchanged from that job (purely structural / architecture-shape-identical,
unaffected by which weights are loaded).

checkpoint_best.pth arrives via the repo's git-ignore-safe transfer convention as
checkpoint_best.pth.txt (byte-identical copy); rename it to checkpoint_best.pth before running
this script if that hasn't been done yet (see /memories/repo/s12_dense_256_u4_analytical_v1_build.md).

Usage (repo root; local .venv has torch+brevitas, no training container needed for export):
    .venv\\Scripts\\python.exe hardware/builds/S12_dense_256_u4_analytical_v1/finn_export_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep.py --skip-cross-test

Output: hardware/builds/S12_dense_256_u4_analytical_v1/outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep.onnx
Then, inside the FINN container:
    docker cp hardware/builds/S12_dense_256_u4_analytical_v1/outputs/quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
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
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))

from finn_export_S12_dense_256_u4_analytical_v1_finn_calibrated import (  # noqa: E402
    DATASET_NAME,
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    DECODER_TYPE,
    DEFAULT_BITS_FILE,
    DEFAULT_PREPROCESSED_DIR,
    NNUNET_RAW,
    load_layer_bits,
)
import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
from finn_export_12_dense_relu_nearest_conv_upsample_256_trained import (  # noqa: E402
    load_calibration_images,
    calibrate_runtime_stats,
    run_finn_inference,
    evaluate,
    compare_to_real_predictions,
)

finn_enet_prod_export.OUT_DIR = Path(__file__).resolve().parent / "outputs"

NET_NAME = (
    "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
    "12_dense_relu_nearest_upsample_256_uniform_int6_perlayer_ft15ep"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / DATASET_NAME
    / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_REAL_PRED_DIR = NNUNET_RAW / f"labelsPr_{NET_NAME}"
NET_NAME_SHORT = "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in"


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
                         help="fine-tuned checkpoint already has its own runtime-stats activation scales "
                              "from training -- skipped by default; pass this to force an EXTRA pass")
    parser.add_argument("--skip-verify", action="store_true",
                         help="skip the mandatory post-export Dice/agreement gate (hardware/verify_export.py)")
    parser.add_argument("--min-fg-dice", type=float, default=0.5,
                         help="gate: minimum foreground Dice of the PyTorch model on the val split "
                              "(ft15ep INT6 measures 0.638)")
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

    print(f"\n=== Building REAL-weight (fine-tuned 15ep), per-layer-bit-width FINN-safe "
          f"S12_dense_256_u4_analytical_v1 -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        checkpoint_path, layer_weight_bits, layer_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE, uint8_input=True,
    ).eval()

    if args.calibrate:
        print("\n=== Calibrating runtime-stats activation scales on real training data (EXTRA pass) ===")
        n_calib = None if args.calibration_images < 0 else args.calibration_images
        calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
        calibrate_runtime_stats(model, calibration_images)
    else:
        print("\n=== Skipping calibration -- fine-tuned checkpoint already carries its own "
              "runtime-stats activation scales from training ===")

    if not args.skip_export:
        print("\n=== Forward-pass sanity check + QONNX export ===")
        # hardware input contract: u = clip(round(z/s_in)+128, 0, 255) as UINT8-valued float
        dummy = torch.randint(0, 256, (1, args.in_channels, 256, 256)).float()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape[2:] == (256, 256), f"output HxW {tuple(out.shape[2:])} != (256,256)"
        assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
        print(f"  forward OK: output shape {tuple(out.shape)}")
        name = "quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in"
        onnx_path = export_model(model, name, dummy, force_input_dtype="UINT8")
        print("\nExported. Copy to FINN container with:")
        print(f"  docker cp hardware/builds/S12_dense_256_u4_analytical_v1/outputs/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")

    if not args.skip_verify:
        import verify_export  # noqa: E402
        print("\n=== Post-export verification gate (hardware/verify_export.py) ===")
        ref_npz = finn_enet_prod_export.OUT_DIR / f"{NET_NAME_SHORT}_verify_ref.npz"
        verify_export.build_reference(model, Path(args.preprocessed_dir), ref_npz, min_fg_dice=args.min_fg_dice)
        if not args.skip_export:
            verify_export.check_onnx(ref_npz, onnx_path, "qonnx_export")
        print("\nVERIFY OK. After the FINN preamble, also run (container):")
        print(f"  python3 verify_export.py --ref {ref_npz.name} --onnx streamline=<preamble>/intermediate_models/step_enet_streamline.onnx "
              f"--first-thresholds <preamble>/intermediate_models/step_enet_streamline.onnx")
    else:
        print("\n  WARNING: --skip-verify given -- this export has NOT been Dice-checked.")


if __name__ == "__main__":
    main()
