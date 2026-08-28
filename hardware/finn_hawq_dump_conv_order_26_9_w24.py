"""Same as finn_hawq_dump_conv_order_26_5_w24.py, but for the REAL PTQ-
calibrated 26_9_w24_s14w12_nonneg_block export (FINNQuantENet26_9_w24HAWQ,
from finn_export_26_9_w24_hawq_joint_ptq.py). Weight values don't matter
for this dump (only module order/type/shape), so this rebuilds the model
fresh rather than re-loading the checkpoint.

Usage (run inside the pytorch training container):
    python hardware/finn_hawq_dump_conv_order_26_9_w24.py

Output: hardware/outputs/finn_exports/quantEnet_26_9_w24_hawq_joint_ptq_int8_conv_order.json
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

from finn_export_26_9_w24_hawq_joint_ptq import (  # noqa: E402
    FINNQuantENet26_9_w24HAWQ,
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
    model = FINNQuantENet26_9_w24HAWQ(
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

    out_path = OUT_DIR / "quantEnet_26_9_w24_hawq_joint_ptq_int8_conv_order.json"
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
