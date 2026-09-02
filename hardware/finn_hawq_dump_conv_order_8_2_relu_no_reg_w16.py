"""Dump an ORDERED list of (logical_name, module_type, weight_shape) for
every weight-bearing / pool node in the HAWQ per-block FINN-safe
8_2_relu_no_reg_w16 model (`FINNQuantENet8w16HAWQ`, from
finn_export_8_2_relu_no_reg_w16_acc2x_hawq_dummy.py), in `named_modules()`
registration order -- see hardware/finn_hawq_dump_conv_order.py (the S19
original this mirrors) for the full rationale. Byte-for-byte copy of
finn_hawq_dump_conv_order_8_2_relu_no_reg_w20.py with only the model class/
block-bits-file/output-name swapped for w16.

Usage (run inside the pytorch training container):
    python hardware/finn_hawq_dump_conv_order_8_2_relu_no_reg_w16.py

Output: hardware/outputs/finn_exports/quantEnet_8_2_relu_no_reg_w16_acc2x_hawq_dummy_conv_order.json
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

from finn_export_8_2_relu_no_reg_w16_acc2x_hawq_dummy import (  # noqa: E402
    FINNQuantENet8w16HAWQ,
    DEFAULT_BLOCK_BITS_FILE,
    OUT_DIR,
)

WEIGHT_MODULE_TYPES = (qnn.QuantConv2d, qnn.QuantConvTranspose2d, nn.MaxPool2d)


def main() -> None:
    with open(DEFAULT_BLOCK_BITS_FILE) as f:
        block_bits = json.load(f)

    torch.manual_seed(0)
    model = FINNQuantENet8w16HAWQ(
        block_bits["stage_weight_bits"], block_bits["stage_act_bits"],
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

    out_path = OUT_DIR / "quantEnet_8_2_relu_no_reg_w16_acc2x_hawq_dummy_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
