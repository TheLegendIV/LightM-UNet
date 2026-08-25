"""Dump an ORDERED list of (logical_name, module_type, weight_shape) for
every weight-bearing / pool node in the HAWQ per-block FINN-safe S19 model
(`FINNQuantENetS19BlockHAWQ`, from finn_export_s19_hawq_block.py), in
`named_modules()` registration order (== forward()/trace order for every
FINN-safe block in this repo, since each block assigns `self.reduce`,
`self.conv`, `self.expand` etc. as attributes in exactly the order they're
invoked in `forward()`).

This sidecar JSON is the PyTorch-side half of the folding-config bridge:
matched up (by validated correspondence, not blind position-zip) against
the ORDERED list of MVAU/VVAU/StreamingMaxPool/ConvolutionInputGenerator
node names FINN produces after `step_specialize_layers`, to translate
`compression/hawq/folding_block_s19.json`'s logical-layer-name keys into
FINN's generated node names for `apply_folding_config`.

Usage (run inside the pytorch training container):
    python hardware/finn_hawq_dump_conv_order.py

Output: hardware/outputs/finn_exports/quantEnet_s19_hawq_block_int8_conv_order.json
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

from finn_export_s19_hawq_block import (  # noqa: E402
    FINNQuantENetS19BlockHAWQ,
    DEFAULT_BLOCK_BITS_FILE,
    DEFAULT_SLOPE_MAP_FILE,
    OUT_DIR,
)

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def main() -> None:
    with open(DEFAULT_BLOCK_BITS_FILE) as f:
        block_bits = json.load(f)
    with open(DEFAULT_SLOPE_MAP_FILE) as f:
        leaky_slope_map = json.load(f)

    torch.manual_seed(0)
    model = FINNQuantENetS19BlockHAWQ(
        block_bits["stage_weight_bits"], block_bits["stage_act_bits"],
        leaky_slope_map=leaky_slope_map,
    ).eval()

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

    out_path = OUT_DIR / "quantEnet_s19_hawq_block_int8_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
