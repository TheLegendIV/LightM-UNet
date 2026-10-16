"""Export the FINE-TUNED (15 epochs, UNIFORM INT8, calibrated warm start) S12-dense 512x512 network with the BILINEAR decoder
(decoder_type="upsample_conv": frozen nearest upsample + frozen depthwise 3x3 INT8 tent-kernel conv on the main
branch of up4/up5) as a QONNX file with the UINT8 input contract (u = clip(round(z/s_in)+128, 0, 255)), plus its conv_order.json.

Checkpoint: Dataset509_ARCADE_1x1_4c / nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_
12_dense_relu_warmstart150ep_bilinear_uniform_int8_perlayer_ft15ep (LayerQuantEnetFINN, U4 widths (4,16,32,16,4), bottlenecks
(4,8,8,2,1), dense_dilation, 5 output classes). Ships as checkpoint_best.pth.txt (git-ignore-safe).

Bits: MILP/artifacts/S12_dense_bilinear_512_uniform_v1/layer_bits_SITES_uniform_int8.json (85 weight / 110 act sites, all 8) = the
per-site schema of the folding json this build is folded from (MILP/artifacts/S12_512_bilin_int8_maxfps/simfifo_pairs, which is
also uniform 8 on all 88 layers/sites).

Usage (repo root, torch+brevitas+nnunetv2 env, e.g. the CPU pytorch container):
    python hardware/builds/S12_dense_512_bilinear_int8_maxfps_v1/finn_export_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep.py [--check-keys]

Output: hardware/builds/S12_dense_512_bilinear_int8_maxfps_v1/outputs/quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_u8in.onnx
        .../outputs/quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_conv_order.json
        .../outputs/quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_u8in_verify_ref.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))

import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
from s12_export import dump_conv_order, load_layer_bits  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
finn_enet_prod_export.OUT_DIR = OUT_DIR

HW = 512
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"
CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DECODER_TYPE = "upsample_conv"
NET_NAME = (
    "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_"
    "12_dense_relu_warmstart150ep_bilinear_uniform_int8_perlayer_ft15ep"
)
_CKPT_DIR = REPO_ROOT / "data" / "nnUNet_results" / DATASET_NAME / f"{NET_NAME}__nnUNetPlans__2d" / "fold_0"
DEFAULT_CHECKPOINT = _CKPT_DIR / ("checkpoint_best.pth" if (_CKPT_DIR / "checkpoint_best.pth").is_file() else "checkpoint_best.pth.txt")
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_bilinear_512_uniform_v1" / "layer_bits_SITES_uniform_int8.json"
)
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "data" / "nnUNet_preprocessed" / DATASET_NAME / "nnUNetPlans_2d"
NAME_STEM = "quantEnet_S12_dense_512_bilinear_int8_maxfps_v1"
NET_NAME_SHORT = f"{NAME_STEM}_ft15ep_u8in"


def check_keys(checkpoint_path: Path, model: torch.nn.Module) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    src = ckpt["network_weights"]
    model_sd = model.state_dict()
    model_params = dict(model.named_parameters(remove_duplicate=False))
    target_keys = set(model_sd) | set(model_params)
    only_src = sorted(k for k in src if k not in target_keys)
    only_model = sorted(k for k in model_sd if k not in src)
    bad_shape = sorted(
        k for k in src
        if k in target_keys and tuple((model_sd[k] if k in model_sd else model_params[k]).shape) != tuple(src[k].shape)
    )
    print(f"checkpoint keys {len(src)}, model state_dict keys {len(model_sd)}, trainer_name={ckpt.get('trainer_name')}")
    print(f"  in checkpoint, not in model : {len(only_src)} {only_src[:10]}")
    print(f"  in model, not in checkpoint : {len(only_model)} {only_model[:10]}")
    print(f"  shape mismatches            : {len(bad_shape)} {bad_shape[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--check-keys", action="store_true", help="only compare checkpoint vs model keys/shapes, then exit")
    parser.add_argument("--skip-verify", action="store_true",
                         help="skip the mandatory post-export Dice/agreement gate (hardware/verify_export.py)")
    parser.add_argument("--min-fg-dice", type=float, default=0.5,
                         help="gate: minimum foreground Dice of the PyTorch model on the val split")
    parser.add_argument("--n-cases", type=int, default=200, help="val cases for the PyTorch reference (lower = faster on CPU)")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    weight_names, act_names = layer_names_for(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)
    assert set(layer_weight_bits.values()) | set(layer_act_bits.values()) == {8}, "this job is uniform INT8"
    arch = dict(
        in_channels=args.in_channels, out_channels=args.out_channels, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN, decoder_type=DECODER_TYPE,
    )

    if args.check_keys:
        check_keys(checkpoint_path, LayerQuantEnetFINN(layer_weight_bits, layer_act_bits, **arch))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dump_conv_order(
        LayerQuantEnetFINN(layer_weight_bits, layer_act_bits, **arch).eval(), args.in_channels,
        OUT_DIR / f"{NAME_STEM}_conv_order.json",
    )

    print(f"\n=== Building REAL-weight (fine-tuned 15ep, uniform INT8) bilinear-decoder S12_dense_512 -- "
          f"{len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(checkpoint_path, layer_weight_bits, layer_act_bits, uint8_input=True, **arch).eval()

    if not args.skip_export:
        print("\n=== Forward-pass sanity check + QONNX export ===")
        dummy = torch.randint(0, 256, (1, args.in_channels, HW, HW)).float()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape[2:] == (HW, HW), f"output HxW {tuple(out.shape[2:])} != ({HW},{HW})"
        assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
        print(f"  forward OK: output shape {tuple(out.shape)}")
        onnx_path = export_model(model, NET_NAME_SHORT, dummy, force_input_dtype="UINT8")
        print(f"\nExported {onnx_path}")

    if not args.skip_verify:
        import verify_export  # noqa: E402
        print("\n=== Post-export verification gate (hardware/verify_export.py) ===")
        ref_npz = OUT_DIR / f"{NET_NAME_SHORT}_verify_ref.npz"
        verify_export.build_reference(model, Path(args.preprocessed_dir), ref_npz, n_cases=args.n_cases,
                                      min_fg_dice=args.min_fg_dice, hw=HW)
        if not args.skip_export:
            verify_export.check_onnx(ref_npz, onnx_path, "qonnx_export")
        print("\nVERIFY OK.")
    else:
        print("\n  WARNING: --skip-verify given -- this export has NOT been Dice-checked.")


if __name__ == "__main__":
    main()
