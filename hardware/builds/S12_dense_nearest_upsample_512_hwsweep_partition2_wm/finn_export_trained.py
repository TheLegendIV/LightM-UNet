"""Trained-weight export for the S12_dense_nearest_upsample_512 dsr/pbi-ratio
hardware-validation sweep (MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/).

Sibling of ../12_dense_relu_nearest_upsample_512/finn_export_12_dense_relu_nearest_upsample_dummy.py,
but loads REAL trained weights via LayerQuantEnetFINN.from_pretrained() from
the one shared checkpoint below (already fine-tuned for this exact per-layer
joint alpha=1.0 candidatebits468 MILP family), instead of torch.manual_seed(0)
dummy weights -- and is parametrized by --tag/--bits-file so one script
serves all 5 sweep points (the "auto-fold" extra baseline build reuses
baseline_both_off's own onnx unchanged; auto-fold only changes the later OOC
build step, not this export).

Checkpoint (shared across all --tag values -- same real trained weights,
re-quantized per each tag's own layer_weight_bits/layer_act_bits):
    data/nnUNet_results/Dataset509_ARCADE_1x1_4c/
    nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_
    12_dense_relu_nearest_upsample_wm_joint_alpha1.0_perlayer_candidatebits468_
    forcedsp_lut50_bram50_dsp90_ft15ep__nnUNetPlans__2d/fold_0/checkpoint_best.pth.txt
    (pulled in manually, renamed .pth.txt to dodge .gitignore -- torch.load()
    doesn't care about the extension).

Usage (inside the pytorch training container):
    python finn_export_trained.py --tag baseline_both_off \\
        --bits-file MILP/artifacts/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/baseline_both_off/layer_bits_SITES_baseline_both_off.json

Output: hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs/
    quantEnet_12_dense_relu_nearest_upsample_trained_<tag>_512x512.onnx
Then, inside the FINN container:
    docker cp <onnx> <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_upsample_512"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import finn_enet_prod_export  # noqa: E402
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
from finn_export_12_dense_relu_nearest_upsample_dummy import (  # noqa: E402
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    DECODER_TYPE,
    load_layer_bits,
)

# Redirect export_model()'s output into this job's own outputs/ folder.
finn_enet_prod_export.OUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / (
        "nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_nearest_upsample_perlayer_"
        "12_dense_relu_nearest_upsample_wm_joint_alpha1.0_perlayer_candidatebits468_"
        "forcedsp_lut50_bram50_dsp90_ft15ep__nnUNetPlans__2d"
    )
    / "fold_0" / "checkpoint_best.pth.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="sweep point name, e.g. baseline_both_off")
    parser.add_argument("--bits-file", required=True, help="that sweep point's own layer_bits_SITES_<tag>.json")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    h, w = args.input_hw
    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building REAL-weight (checkpoint={checkpoint_path.name}, tag={args.tag!r}) "
          f"12_dense_relu_nearest_upsample -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN.from_pretrained(
        checkpoint_path, layer_weight_bits, layer_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE,
    ).eval()

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = f"quantEnet_12_dense_relu_nearest_upsample_trained_{args.tag}_{h}x{w}"
    export_model(model, name, dummy)
    print(f"\nExported: hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs/{name}.onnx")
    print("Copy to FINN container with:")
    print(f"  docker cp hardware/builds/S12_dense_nearest_upsample_512_hwsweep_partition2_wm/outputs/{name}.onnx "
          "<finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
