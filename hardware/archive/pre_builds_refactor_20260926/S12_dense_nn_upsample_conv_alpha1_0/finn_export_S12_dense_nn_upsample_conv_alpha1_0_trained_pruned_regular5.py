"""Export a FINN-compatible, PER-LAYER HAWQ mirror of the REAL, QAT
fine-tuned nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_
joint_alpha1.0_candidatebits468_lut50_bram20_dsp90_maxlat200ms_joinbalance1.1_
calibrated checkpoint, with REAL TRAINED WEIGHTS transferred wherever the
FINN-safe topology is structurally identical to the real LayerQuantENet --
IDENTICAL to finn_export_S12_dense_nn_upsample_conv_alpha1_0_trained.py
except `regular5.0` (stage5's one regular bottleneck) is PRUNED out of the
graph entirely, per explicit instruction that this block is useless.

Pruning mechanism: BOTTLENECKS_PER_STAGE's n5 is set to 0 instead of 1.
`_make_layer_shallow_stage(channels, n_ops=0, ...)` (see
enet/nnunetv2/nets/LayerQuantENet.py) returns an empty `nn.Sequential()`
for `self.regular5`, which is a no-op passthrough in forward() -- so this
is a genuine architectural removal (zero regular5.0.* nodes ever get
constructed/traced/exported), not a post-hoc ONNX graph edit. The real
checkpoint was trained WITH regular5.0 present, so its regular5.0.*
weights simply have no matching key in this pruned model's state_dict and
are silently dropped by the existing strict=False key+shape filter in
load_real_weights -- no code change needed there. layer_names_for(...)
(used for both the per-layer bits lookup and the conv-order dump) also
naturally omits every regular5.0.* site name once n5=0, so the bits JSON
(still keyed for the un-pruned architecture) simply has a few unused
entries -- harmless.

Usage (run inside the pytorch training container):
    docker exec elegant_cannon python /workspace/LightM-UNet/hardware/finn_export_S12_dense_nn_upsample_conv_alpha1_0_trained_pruned_regular5.py --input-hw 512 512

Output: hardware/outputs/finn_exports/quantEnet_S12_dense_nn_upsample_conv_alpha1.0_trained_pruned_regular5_int8_512x512.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_S12_dense_nn_upsample_conv_alpha1.0_trained_pruned_regular5_int8_512x512.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402
from finn_export_12_dense_relu_warmstart150ep_alpha025_trained import (  # noqa: E402
    load_real_weights, calibrate_runtime_stats, load_calibration_images,
)
from finn_export_S12_dense_nn_upsample_conv_alpha1_0_dummy import (  # noqa: E402
    load_layer_bits, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, DEFAULT_BITS_FILE,
)

# Only architectural delta vs. the un-pruned trained sibling: n5 (regular5's
# bottleneck count) is 0 instead of 1.
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 0)

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerLayerQuantENet_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_"
      "lut50_bram20_dsp90_maxlat200ms_joinbalance1.1_calibrated__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_PREPROCESSED_DIR = (
    REPO_ROOT / "data" / "nnUNet_preprocessed" / "Dataset509_ARCADE_1x1_4c" / "nnUNetPlans_2d"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument(
        "--calibration-images", type=int, default=-1,
        help="number of real training patches to calibrate on; -1 (default) uses the full train split",
    )
    parser.add_argument(
        "--argmax-in-pl", action="store_true",
        help="append a channel-wise top-1 (ONNX TopK k=1) after `final` so FINN's "
             "InferLabelSelectLayer can lower it to an in-PL LabelSelect HW op instead of "
             "exporting raw per-class logits for CPU-side argmax.",
    )
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, decoder_type=DECODER_TYPE, use_dilated=True, use_asymmetric=False,
        use_strided=True, use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False,
        separable_dilated=False,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building REAL-weight, per-layer-bit-width FINN-safe S12_dense_nn_upsample_conv "
          f"(alpha=1.0, regular5.0 PRUNED) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE, argmax_output=args.argmax_in_pl,
    ).eval()
    assert len(model.regular5) == 0, "expected regular5 to be pruned to an empty nn.Sequential (n5=0)"

    print("\n=== Loading real trained checkpoint (regular5.0.* keys silently dropped) ===")
    load_real_weights(model, checkpoint_path)

    print("\n=== Calibrating runtime-stats activation scales on real training data ===")
    n_calib = None if args.calibration_images < 0 else args.calibration_images
    calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
    calibrate_runtime_stats(model, calibration_images)

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    expected_out_ch = 1 if args.argmax_in_pl else args.out_channels
    assert out.shape[1] == expected_out_ch, f"output channels {out.shape[1]} != {expected_out_ch}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_S12_dense_nn_upsample_conv_alpha1.0_trained_pruned_regular5_int8"
    if (h, w) != (64, 64):
        name += f"_{h}x{w}"  # distinct filename -- never overwrite the reference 64x64 export
    if args.argmax_in_pl:
        name += "_argmaxpl"
    export_model(model, name, dummy, force_output_dtype=None if args.argmax_in_pl else "INT8")

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
