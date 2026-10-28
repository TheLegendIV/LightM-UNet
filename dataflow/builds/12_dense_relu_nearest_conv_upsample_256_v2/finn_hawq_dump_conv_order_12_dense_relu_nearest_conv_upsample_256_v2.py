"""v2 sibling of finn_hawq_dump_conv_order_12_dense_relu_nearest_conv_upsample_256.py
-- purely structural (module call order), architecture-identical to v1, so
this would in fact produce a byte-identical conv_order.json to v1's; kept as
its own script only so the v2 export's output name has a matching dump
script per this repo's naming convention (see AGENTS.md), pointing at v2's
NET_NAME/bits file via finn_export_..._v2_trained's constants instead of
v1's.

Usage (run inside the pytorch training container):
    python hardware/builds/12_dense_relu_nearest_conv_upsample_256_v2/finn_hawq_dump_conv_order_12_dense_relu_nearest_conv_upsample_256_v2.py

Output: hardware/builds/12_dense_relu_nearest_conv_upsample_256_v2/outputs/
    quantEnet_12_dense_relu_nearest_conv_upsample_256_v2_trained_conv_order.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(REPO_ROOT / "hardware"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# v1's job folder: only for the shared helpers re-exported by the v2 export module.
sys.path.insert(0, str(REPO_ROOT / "hardware" / "builds" / "12_dense_relu_nearest_conv_upsample_256"))

from finn_export_12_dense_relu_nearest_conv_upsample_256_v2_trained import (  # noqa: E402
    LayerQuantEnetFINN,
    load_layer_bits,
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    DECODER_TYPE,
    DEFAULT_BITS_FILE,
)
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def main() -> None:
    shape_kwargs = dict(
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(DEFAULT_BITS_FILE, weight_names, act_names)

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

    out_path = OUT_DIR / "quantEnet_12_dense_relu_nearest_conv_upsample_256_v2_trained_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
