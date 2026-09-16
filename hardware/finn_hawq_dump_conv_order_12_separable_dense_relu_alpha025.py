"""Dump an ORDERED list of (logical_name, module_type, weight_shape) for
every weight-bearing / pool node in the PER-LAYER HAWQ, alpha=0.25
12_separable_dense_relu FINN-safe model (`LayerQuantEnetFINN`, from
finn_export_12_separable_dense_relu_alpha025_dummy.py), in
forward-execution order -- see
hardware/finn_hawq_dump_conv_order_12_dense_relu_warmstart150ep_alpha025.py
(the DENSE, non-separable sibling this mirrors) for the full rationale.
This ordered list is the positional bridge between the FINN dataflow
graph's weight-like nodes (MVAU/VVAU/MaxPool, in forward-execution order)
and this architecture's real per-layer folding config
(compression/hawq/artifacts/S12_ILP_outputs_perlayer_forcedsp_lut70/
layer_bits_folding_12_separable_dense_relu_joint_alpha0.25_
candidatebits468_forcedsp_lut70.json's "per_layer" dict, keyed by these
same logical dotted names -- confirmed this file already has separate
"stage2.N.conv.0"/"stage2.N.conv.3" keys for every dilated block, matching
this architecture's real (k,1)+(1,k)-split site names exactly, no extra
bridging needed for that split beyond what the dense sibling's bridge
script already does generically).

Byte-for-byte copy of finn_hawq_dump_conv_order_12_dense_relu_
warmstart150ep_alpha025.py with only the imported model/bits-file module
swapped to this file's own separable_dilated=True variant. Same hook-based
(register_forward_hook, not a static named_modules() walk) approach --
needed because this "dense_dilation" context pattern's stage2/stage3 blocks
fork their input and apply the SAME shared conv module to two different
branch copies (named_modules() would only report such a shared module
once, silently under-counting vs. the real graph's node count).

Usage (run inside the pytorch training container):
    python hardware/finn_hawq_dump_conv_order_12_separable_dense_relu_alpha025.py

Output: hardware/outputs/finn_exports/
    quantEnet_12_separable_dense_relu_alpha025_dummy_int8_conv_order.json
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

from finn_export_12_separable_dense_relu_alpha025_dummy import (  # noqa: E402
    LayerQuantEnetFINN,
    load_layer_bits,
    CHANNELS,
    BOTTLENECKS_PER_STAGE,
    CONTEXT_PATTERN,
    SEPARABLE_DILATED,
    DEFAULT_BITS_FILE,
    OUT_DIR,
)
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def main() -> None:
    shape_kwargs = dict(
        out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False,
        separable_dilated=SEPARABLE_DILATED,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(DEFAULT_BITS_FILE, weight_names, act_names)

    torch.manual_seed(0)
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=1, out_channels=5,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED,
    ).eval()

    # See module docstring / the dense sibling's own identical comment:
    # a static named_modules() walk under-counts shared (forked) modules.
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

    out_path = OUT_DIR / "quantEnet_12_separable_dense_relu_alpha025_dummy_int8_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
