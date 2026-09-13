"""Dump an ORDERED list of (logical_name, module_type, weight_shape) for
every weight-bearing / pool node in the PER-LAYER HAWQ, alpha=0.25
12_dense_relu_warmstart150ep FINN-safe model (`FINNQuantENet`, from
finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py), in
named_modules() registration order -- see
hardware/finn_hawq_dump_conv_order_12_separable_dense_relu.py (the
per-BLOCK-HAWQ sibling this mirrors) for the full rationale. This ordered
list is the positional bridge between the FINN dataflow graph's weight-like
nodes (MVAU/VVAU/MaxPool, in forward-execution order) and this
architecture's real per-layer folding config
(compression/hawq/artifacts/12_dense_relu_warmstart150ep_ILP_outputs_
perlayer_forcedsp_lut70/layer_bits_folding_..._alpha0.25_...json's
"per_layer" dict, keyed by these same logical dotted names).

Usage (run inside the pytorch training container):
    python hardware/finn_hawq_dump_conv_order_12_dense_relu_warmstart150ep_alpha025.py

Output: hardware/outputs/finn_exports/
    quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8_conv_order.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finn_export_12_dense_relu_warmstart150ep_alpha025_dummy import (  # noqa: E402
    LayerQuantEnetFINN,
    load_layer_bits,
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    DEFAULT_BITS_FILE,
    OUT_DIR,
)
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def main() -> None:
    shape_kwargs = dict(
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(DEFAULT_BITS_FILE, weight_names, act_names)

    torch.manual_seed(0)
    model = LayerQuantEnetFINN(layer_weight_bits, layer_act_bits, in_channels=1, out_channels=5).eval()

    ordered = []
    for name, mod in model.named_modules():
        if isinstance(mod, WEIGHT_MODULE_TYPES):
            kind = type(mod).__name__
            shape = None
            if hasattr(mod, "weight") and mod.weight is not None:
                shape = list(mod.weight.shape)
            ordered.append({"logical_name": name, "module_type": kind, "weight_shape": shape})

    print(f"Found {len(ordered)} weight-bearing/pool modules in named_modules() order:")
    for entry in ordered:
        print(f"  {entry['logical_name']:35s} {entry['module_type']:20s} {entry['weight_shape']}")

    out_path = OUT_DIR / "quantEnet_12_dense_relu_warmstart150ep_alpha025_dummy_int8_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
