"""Export a FINN-compatible, PER-LAYER HAWQ mirror of
nnUNetTrainerENet_12_separable_dense_relu (separable (k,1)+(1,k)-factored
dilated context, SEPARABLE_DILATED=True, plain ReLU, alpha=0.25 per-layer
HAWQ bit-width assignment -- see
compression/hawq/config_12_separable_dense_relu.py and
compression/hawq/artifacts/S12_ILP_outputs_perlayer_forcedsp_lut70/
layer_bits_SITES_12_separable_dense_relu_joint_alpha0.25_candidatebits468_
forcedsp_lut70.json -- NOT the sibling layer_bits_folding_*.json, whose
layer_weight_bits/layer_act_bits use a DIFFERENT, coarser conv-input-
activation schema; see memories/repo/finn_12_dense_relu_alpha025_perlayer.md
for the full distinction, which applies unchanged to this sibling family).

Byte-for-byte copy of finn_export_12_dense_relu_warmstart150ep_alpha025_
dummy.py (the DENSE, non-separable sibling this mirrors) with only
CHANNELS/BOTTLENECKS_PER_STAGE (numerically identical, but re-stated here
for this file's own clarity)/DEFAULT_BITS_FILE/separable_dilated=True
changed -- the FINN-safe model itself (FINNInitialBlockConcat/
FINNDownsamplingBottleneck/FINNUpsamplingBottleneck/LayerQuantEnetFINN, and
the 4 substitutions vs. the real LayerQuantENet) is the SAME shared class in
enet/nnunetv2/nets/LayerQuantEnetFINN.py -- that file's own
`separable_dilated` constructor kwarg (added for this task) threads straight
through to stage2/stage3's `_make_layer_context_stage` calls; no
architecture-level code duplicated here.

DUMMY variant: fresh (torch.manual_seed(0)) conv/BN weights throughout --
validates the graph/export pipeline structurally, and validates that the
real per-layer bits JSON's key set lines up exactly with this exact
architecture's real site names (in particular: stage2/stage3's dilated
blocks split into "conv.0"/"conv.3" sites instead of dense's single "conv"
-- see LayerQuantRegularBottleneck's own docstring), before loading the
real checkpoint (see finn_export_12_separable_dense_relu_alpha025_trained.py,
which reuses this file's load_layer_bits verbatim).

Usage (run inside the pytorch training container):
    docker exec <container> python /workspace/LightM-UNet/hardware/finn_export_12_separable_dense_relu_alpha025_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_alpha025_dummy_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_alpha025_dummy_int8.onnx \\
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
from finn_export_s13_leaky_frozen import export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)          # initial, s1, s23 (shared), s4, s5 -- same as 12_dense_relu_warmstart150ep
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
SEPARABLE_DILATED = True
DEFAULT_BITS_FILE = (
    REPO_ROOT / "compression" / "hawq" / "artifacts"
    / "S12_ILP_outputs_perlayer_forcedsp_lut70"
    / "layer_bits_SITES_12_separable_dense_relu_joint_alpha0.25_candidatebits468_forcedsp_lut70.json"
)
FALLBACK_BITS = 6  # same user decision as the dense sibling -- any site missing from the real JSON falls back to 6, not 8


def load_layer_bits(
    bits_file: Path, weight_names: tuple[str, ...], act_names: tuple[str, ...], fallback: int = FALLBACK_BITS,
) -> tuple[dict[str, int], dict[str, int]]:
    """Loads a layer_bits_SITES_*.json (fine-grained per-activation-SITE
    schema -- NOT the sibling layer_bits_folding_*.json, see module
    docstring) and expands it to a complete dict covering every real site
    this exact architecture needs, falling back to `fallback` for any site
    missing from the file (never crashes on a missing key, unlike
    LayerQuantENet's own strict validation)."""
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
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False,
        separable_dilated=SEPARABLE_DILATED,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building fresh-weight, per-layer-bit-width FINN-safe 12_separable_dense_relu "
          f"(alpha=0.25) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED,
    ).eval()

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_12_separable_dense_relu_alpha025_dummy_int8"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
