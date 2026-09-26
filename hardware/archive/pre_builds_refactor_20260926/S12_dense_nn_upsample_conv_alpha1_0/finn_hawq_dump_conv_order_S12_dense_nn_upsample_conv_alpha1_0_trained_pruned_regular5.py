"""Dump an ORDERED list of (logical_name, module_type, weight_shape) for
every weight-bearing / pool node in the PER-LAYER HAWQ, alpha=1.0
S12_dense_nn_upsample_conv FINN-safe model, WITH `regular5.0` pruned out
of the graph (BOTTLENECKS_PER_STAGE n5=0) -- see
finn_export_S12_dense_nn_upsample_conv_alpha1_0_trained_pruned_regular5.py
for the pruning rationale. Only difference vs. the un-pruned sibling
(finn_hawq_dump_conv_order_12_dense_relu_nearest_conv_upsample.py): 3 fewer
entries (regular5.0.reduce.0, regular5.0.conv, regular5.0.expand.0 all
gone). Uses fresh random weights (torch.manual_seed(0)) -- conv order is a
pure function of architecture SHAPE, never of actual weight values, so
this doesn't need (and deliberately skips) the real checkpoint.

This ordered list is the positional bridge between the FINN dataflow
graph's weight-like nodes (MVAU/VVAU/MaxPool, in forward-execution order)
and this architecture's real per-layer folding config
(MILP/artifacts/S12_dense_nn_upsample_v1/layer_bits_folding_..._alpha1.0_..._maxlat200ms_joinbalance1.1.json's
"per_layer" dict, keyed by these same logical dotted names) -- the pruned
regular5.0.* keys simply go unused in that JSON now, harmlessly.

Usage (run inside the pytorch training container):
    docker exec elegant_cannon python /workspace/LightM-UNet/hardware/finn_hawq_dump_conv_order_S12_dense_nn_upsample_conv_alpha1_0_trained_pruned_regular5.py

Output: hardware/outputs/finn_exports/
    quantEnet_S12_dense_nn_upsample_conv_alpha1.0_trained_pruned_regular5_int8_conv_order.json
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

from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_export_S12_dense_nn_upsample_conv_alpha1_0_dummy import (  # noqa: E402
    load_layer_bits, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, DEFAULT_BITS_FILE,
)
from finn_export_S12_dense_nn_upsample_conv_alpha1_0_trained_pruned_regular5 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE,
)

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)
OUT_DIR = REPO_ROOT / "hardware" / "outputs" / "finn_exports"


def main() -> None:
    shape_kwargs = dict(
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
        decoder_type=DECODER_TYPE,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(DEFAULT_BITS_FILE), weight_names, act_names)

    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=1, out_channels=5,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        decoder_type=DECODER_TYPE,
    ).eval()
    assert len(model.regular5) == 0, "expected regular5 to be pruned to an empty nn.Sequential (n5=0)"

    # Forward hooks record one entry PER ACTUAL CALL (true execution
    # order), not a static named_modules() walk -- see sibling scripts'
    # comments for why (fork/join shared-module calls would undercount).
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

    out_path = OUT_DIR / "quantEnet_S12_dense_nn_upsample_conv_alpha1.0_trained_pruned_regular5_int8_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
