"""Export a FINN-compatible, PER-LAYER HAWQ mirror of
nnUNetTrainerENet_12_dense_relu_warmstart150ep (dense KxK dilated context,
SEPARABLE_DILATED=False, plain ReLU, alpha=0.25 per-layer HAWQ bit-width
assignment -- see compression/hawq/config_12_dense_relu_warmstart150ep.py
and compression/hawq/artifacts/12_dense_relu_warmstart150ep_ILP_outputs_
perlayer_forcedsp_lut70/layer_bits_SITES_12_dense_relu_warmstart150ep_joint_
alpha0.25_candidatebits468_forcedsp_lut70.json -- NOT the sibling
layer_bits_folding_*.json, whose layer_weight_bits/layer_act_bits use a
DIFFERENT, coarser conv-input-activation schema; see
memories/repo/finn_12_dense_relu_alpha025_perlayer.md for the full
distinction).

DUMMY variant: fresh (torch.manual_seed(0)) conv/BN weights throughout --
validates the graph/export pipeline structurally, and validates that the
real per-layer bits JSON's key set lines up exactly with this exact
architecture's real site names, before loading the real ft15ep QAT
checkpoint (see finn_export_12_dense_relu_warmstart150ep_alpha025_trained.py,
which reuses this file's load_layer_bits verbatim).

The FINN-safe model itself (FINNInitialBlockConcat/FINNDownsamplingBottleneck/
FINNUpsamplingBottleneck/LayerQuantEnetFINN, and the 4 substitutions they make
vs. the real LayerQuantENet) now lives in
enet/nnunetv2/nets/LayerQuantEnetFINN.py (moved 2026-09-14, see that file's
own module docstring for the full derivation) -- this script only handles
bit-width loading + fresh-weight construction + QONNX export for this
particular config.

Since USE_PRELU=False for this config, the REAL LayerQuantENet's own
activation modules devolve to plain qnn.QuantReLU (`_quant_block_act` with
negative_slope=None -- see QuantENet.py) and its residual_add is a real,
already-FINN-safe qnn.QuantEltwiseAdd (single shared input_quant across
both operands, per LayerQuantRegularBottleneck's own docstring). This means
LayerQuantRegularBottleneck (imported directly from
enet/nnunetv2/nets/LayerQuantENet.py, via the same
_make_layer_shallow_stage/_make_layer_context_stage assembly helpers that
file itself uses) is reused UNMODIFIED here -- no FINN-specific
reimplementation needed. LayerQuantInitialBlock needs one (see
LayerQuantEnetFINN.py's docstring point 0). LayerQuantDSCNoProjectionBottleneck
is never instantiated (USE_DSC=False globally for this config).

Usage (run inside the pytorch training container, e.g. `lightm_pytorch`):
    docker exec lightm_pytorch python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8.onnx \\
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
DEFAULT_BITS_FILE = (
    REPO_ROOT / "compression" / "MILP" / "artifacts"  # moved from compression/hawq/artifacts
    / "12_dense_relu_warmstart150ep_ILP_outputs_perlayer_forcedsp_lut70"
    / "layer_bits_SITES_12_dense_relu_warmstart150ep_joint_alpha0.25_candidatebits468_forcedsp_lut70.json"
)
FALLBACK_BITS = 6  # user decision -- any site missing from the real JSON falls back to 6, not 8


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
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building fresh-weight, per-layer-bit-width FINN-safe 12_dense_relu_warmstart150ep "
          f"(alpha=0.25) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
    ).eval()

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
