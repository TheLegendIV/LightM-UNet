"""Export the FINE-TUNED (15 epochs, uniform INT6) S12-dense 256x256 network with the BILINEAR decoder
(decoder_type="upsample_conv": frozen nearest upsample + frozen depthwise 3x3 INT8 tent-kernel conv on the main
branch of up4/up5) as a QONNX file with the UINT8 input contract (u = clip(round(z/s_in)+128, 0, 255)).

Same architecture as the nearest-upsample ft15ep export (U4 widths (4,16,32,16,4), bottlenecks (4,8,8,2,1),
dense_dilation, Dataset510_ARCADE_256_4c); only the decoder, the checkpoint and the exported name differ. The
per-site bits file is the one the QAT job trained with (site names are identical for both decoder types).

checkpoint_best.pth arrives as checkpoint_best.pth.txt (git-ignore-safe transfer convention); the default
checkpoint path below uses the .txt copy if the .pth is not there.

Usage (repo root, local .venv has torch+brevitas):
    .venv\\Scripts\\python.exe hardware/builds/S12_dense_256_u4_bilinear_analytical_v1/finn_export_S12_dense_256_u4_bilinear_analytical_v1_ft15ep.py

Output: hardware/builds/S12_dense_256_u4_bilinear_analytical_v1/outputs/quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in.onnx
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
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "S12_dense_256_u4_analytical_v1"))
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))

from finn_export_S12_dense_256_u4_analytical_v1_finn_calibrated import (  # noqa: E402
    DATASET_NAME,
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    DEFAULT_PREPROCESSED_DIR,
    NNUNET_RAW,
    load_layer_bits,
)
import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402

finn_enet_prod_export.OUT_DIR = Path(__file__).resolve().parent / "outputs"

DECODER_TYPE = "upsample_conv"
NET_NAME = (
    "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_"
    "12_dense_relu_bilinear_upsample_256_uniform_int6_perlayer_ft15ep"
)
_CKPT_DIR = REPO_ROOT / "data" / "nnUNet_results" / DATASET_NAME / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0"
DEFAULT_CHECKPOINT = _CKPT_DIR / ("checkpoint_best.pth" if (_CKPT_DIR / "checkpoint_best.pth").is_file() else "checkpoint_best.pth.txt")
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_bilinear_256_uniform_int6_v1" / "layer_bits_SITES_uniform_int6.json"
)
NET_NAME_SHORT = "quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--skip-verify", action="store_true",
                         help="skip the mandatory post-export Dice/agreement gate (hardware/verify_export.py)")
    parser.add_argument("--min-fg-dice", type=float, default=0.5,
                         help="gate: minimum foreground Dice of the PyTorch model on the val split")
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

    print(f"\n=== Building REAL-weight (fine-tuned 15ep) bilinear-decoder S12_dense_256_u4 -- "
          f"{len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        checkpoint_path, layer_weight_bits, layer_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE, uint8_input=True,
    ).eval()

    if not args.skip_export:
        print("\n=== Forward-pass sanity check + QONNX export ===")
        dummy = torch.randint(0, 256, (1, args.in_channels, 256, 256)).float()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape[2:] == (256, 256), f"output HxW {tuple(out.shape[2:])} != (256,256)"
        assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
        print(f"  forward OK: output shape {tuple(out.shape)}")
        onnx_path = export_model(model, NET_NAME_SHORT, dummy, force_input_dtype="UINT8")
        print("\nExported. Copy to FINN container with:")
        print(f"  docker cp hardware/builds/S12_dense_256_u4_bilinear_analytical_v1/outputs/{NET_NAME_SHORT}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")

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
