"""Export a FINN-compatible, PER-LAYER HAWQ mirror of
nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample (dense KxK dilated
context, SEPARABLE_DILATED=False, plain ReLU, decoder_type=
"nearest_conv_upsample" -- nearest-neighbor resize + a learned 3x3
conv+BN+act on the decoder's skip/main branch), per-layer HAWQ bit-width
assignment alpha=1.0/bram20/dsp90/maxlat200ms/joinbalance1.1 -- see
MILP/artifacts/S12_dense_nn_upsample_v1/layer_bits_SITES_
12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_
forcedsp_lut50_bram20_dsp90_maxlat200ms_joinbalance1.1.json and
compression/slurm/qat_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_
perlayer_candidatebits468_forcedsp_lut50_bram20_dsp90_maxlat200ms_
joinbalance1.1_ft15ep.job.

DUMMY variant: fresh (torch.manual_seed(0)) conv/BN weights throughout --
validates the graph/export pipeline structurally (including the new
skip_resize_conv branch this decoder_type adds vs. the "upsample_conv"
family), and validates that the real per-layer bits JSON's key set lines up
exactly with this exact architecture's real site names, before the real
ft15ep QAT checkpoint (still training as of this script's creation) is
ready to export for real.

Same pattern as finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py,
just for this architecture -- see that file's own docstring for the full
rationale of the load_layer_bits fallback behavior.

Usage (run inside the pytorch training container):
    docker exec elegant_cannon python /workspace/LightM-UNet/hardware/finn_export_S12_dense_nn_upsample_conv_alpha1_0_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_S12_dense_nn_upsample_conv_alpha1.0_dummy_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_enet_prod_export import export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)          # initial, s1, s23 (shared), s4, s5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DECODER_TYPE = "nearest_conv_upsample"
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_nn_upsample_v1"
    / "layer_bits_SITES_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_"
      "forcedsp_lut50_bram20_dsp90_maxlat200ms_joinbalance1.1.json"
)
FALLBACK_BITS = 6  # same fallback convention as the warmstart150ep dummy export


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

    layer_weight_bits = {n: raw_w.get(n, fallback) for n in weight_names}
    layer_act_bits = {n: raw_a.get(n, fallback) for n in act_names}
    return layer_weight_bits, layer_act_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
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

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, decoder_type=DECODER_TYPE, use_dilated=True, use_asymmetric=False,
        use_strided=True, use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False,
        separable_dilated=False,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building fresh-weight, per-layer-bit-width FINN-safe S12_dense_nn_upsample_conv "
          f"(alpha=1.0) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE, argmax_output=args.argmax_in_pl,
    ).eval()

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    expected_out_ch = 1 if args.argmax_in_pl else args.out_channels
    assert out.shape[1] == expected_out_ch, f"output channels {out.shape[1]} != {expected_out_ch}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_S12_dense_nn_upsample_conv_alpha1.0_dummy_int8"
    if args.argmax_in_pl:
        name += "_argmaxpl"
    export_model(model, name, dummy, force_output_dtype=None if args.argmax_in_pl else "INT8")

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
