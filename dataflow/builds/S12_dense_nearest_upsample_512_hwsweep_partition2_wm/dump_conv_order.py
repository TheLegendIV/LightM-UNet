"""Standalone conv-order dump for this job -- avoids the broken REPO_ROOT/
DEFAULT_BITS_FILE path assumptions in the sibling
../12_dense_relu_nearest_upsample_512/finn_hawq_dump_conv_order_....py
(its REPO_ROOT resolves to hardware/, not the actual repo root, and its
hardcoded DEFAULT_BITS_FILE doesn't exist on disk) -- rather than touch that
shared architecture-level file, this is a self-contained copy of the same
forward-hook logic, using one of THIS job's own valid bits files (bit
VALUES are irrelevant to module CALL ORDER, so any of the 5 tags' own
layer_bits_SITES_<tag>.json works as the structural source).

Usage (inside the pytorch training container):
    python dump_conv_order.py [--bits-file <any tag's layer_bits_SITES_*.json>]

Output: outputs/quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))

from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CONTEXT_PATTERN = "dense_dilation"
DECODER_TYPE = "nearest_upsample"

DEFAULT_BITS_FILE = (
    REPO_ROOT / "MILP" / "artifacts" / "S12_dense_nearest_upsample_512_hwsweep_partition2_wm"
    / "baseline_both_off" / "layer_bits_SITES_baseline_both_off.json"
)

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def load_layer_bits(bits_file: Path, weight_names, act_names):
    with open(bits_file) as f:
        data = json.load(f)
    weight_bits_map = data["layer_weight_bits"]
    act_bits_map = data["layer_act_bits"]
    layer_weight_bits = {name: weight_bits_map[name] for name in weight_names}
    layer_act_bits = {name: act_bits_map[name] for name in act_names}
    return layer_weight_bits, layer_act_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    args = parser.parse_args()

    shape_kwargs = dict(
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=1, out_channels=5,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE,
    ).eval()

    name_by_id: dict[int, str] = {}
    modules_to_hook: list[nn.Module] = []
    for name, mod in model.named_modules():
        if isinstance(mod, WEIGHT_MODULE_TYPES) and id(mod) not in name_by_id:
            name_by_id[id(mod)] = name
            modules_to_hook.append(mod)

    ordered: list[dict] = []

    def _record(mod, _inp, _out):
        kind = type(mod).__name__
        shape = None
        if hasattr(mod, "weight") and mod.weight is not None:
            shape = list(mod.weight.shape)
        ordered.append({"logical_name": name_by_id[id(mod)], "module_type": kind, "weight_shape": shape})

    handles = [mod.register_forward_hook(_record) for mod in modules_to_hook]
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64))
    for h in handles:
        h.remove()

    print(f"Found {len(ordered)} weight-bearing/pool module CALLS in forward-execution order:")
    for entry in ordered:
        print(f"  {entry['logical_name']:35s} {entry['module_type']:20s} {entry['weight_shape']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
