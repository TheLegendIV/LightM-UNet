"""Export a FINN-compatible, PER-LAYER HAWQ mirror of
nnUNetTrainerENet_12_dense_relu_nearest_conv_upsample (dense KxK dilated
context, SEPARABLE_DILATED=False, plain ReLU, decoder_type=
"nearest_conv_upsample" -- nearest-neighbor resize + a REAL learned 3x3
skip_resize_conv on the up4/up5 skip branch, replacing the old
"upsample_conv" bilinear-substitute decoder -- see
MILP/config_12_dense_relu_nearest_conv_upsample.py and
enet/nnunetv2/nets/LayerQuantEnetFINN.py's FINNUpsamplingBottleneck
docstring for the full rationale). Byte-for-byte identical to
finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py otherwise --
same channels/bottleneck depths/context pattern -- only DECODER_TYPE and
the bits-file/output name differ. alpha=1.0, joint per-layer bits+folding
ILP solve (MILP/artifacts/S12_dense_nn_upsample_v1, hard caps LUT 50%/
BRAM 20%/DSP 90%, max-latency 200ms, max-join-imbalance 1.1 -- see that
solve's own run_args.json/summary.csv and
compression/slurm/qat_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_perlayer_candidatebits468_forcedsp_lut50_bram20_dsp90_maxlat200ms_joinbalance1.1_ft15ep.job
for the exact chosen config -- this is the SITES variant, not the folding
one, matching finn_export_..._alpha025_dummy.py's own DEFAULT_BITS_FILE
convention of loading layer_bits_SITES_*.json, NOT layer_bits_folding_*.json).

DUMMY variant: fresh (torch.manual_seed(0)) conv/BN weights throughout --
validates the graph/export pipeline structurally (incl. the new
skip_resize_conv site + nearest-neighbor Resize node) before the real
ft15ep QAT checkpoint lands. Once it lands, copy this file to
finn_export_12_dense_relu_nearest_conv_upsample_trained.py and swap the
fresh LayerQuantEnetFINN(...) construction for
LayerQuantEnetFINN.from_pretrained(checkpoint_path, ...) (same pattern as
finn_export_12_dense_relu_warmstart150ep_alpha025_trained.py vs. its own
_dummy sibling).

Usage (run inside the pytorch training container, e.g. `lightm_pytorch`):
    docker exec lightm_pytorch python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_nearest_conv_upsample_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
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
from finn_enet_prod_export import export_model  # noqa: E402 -- archived finn_export_s13_leaky_frozen's dynamo=False breaks on torch<2.5

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)          # initial, s1, s23 (shared), s4, s5
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DECODER_TYPE = "nearest_conv_upsample"
DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_nn_upsample_v1"
    / "layer_bits_SITES_12_dense_relu_nearest_conv_upsample_joint_alpha1.0_candidatebits468_forcedsp_lut50_bram20_dsp90_maxlat200ms_joinbalance1.1.json"
)
FALLBACK_BITS = 6  # user decision -- any site missing from the real JSON falls back to 6, not 8


def load_layer_bits(
    bits_file: Path, weight_names: tuple[str, ...], act_names: tuple[str, ...], fallback: int = FALLBACK_BITS,
) -> tuple[dict[str, int], dict[str, int]]:
    """Loads a layer_bits_SITES_*.json (fine-grained per-activation-SITE
    schema -- NOT the sibling layer_bits_folding_*.json) and expands it to a
    complete dict covering every real site this exact architecture needs,
    falling back to `fallback` for any site missing from the file (never
    crashes on a missing key)."""
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
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building fresh-weight, per-layer-bit-width FINN-safe 12_dense_relu_nearest_conv_upsample "
          f"(alpha=1.0, decoder_type={DECODER_TYPE!r}) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
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

    name = "quantEnet_12_dense_relu_nearest_conv_upsample_dummy_int8"
    if (h, w) != (64, 64):
        name += f"_{h}x{w}"  # distinct filename -- never overwrite the reference 64x64 export
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
