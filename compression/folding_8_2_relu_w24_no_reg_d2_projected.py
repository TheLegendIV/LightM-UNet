"""Real folding ILP feasibility check for nnUNetTrainerENet_8_2_relu_w24_
no_reg_d2_projected (compression/results.csv's 8_2_relu_w24_projected_
slots_ablation stage, dice=0.768) at uniform INT8 (W8A8) -- "PTQ to int8"
here means evaluating the closed-form FINN cost model at 8-bit weights/
activations network-wide, NOT an actual Brevitas calibration pass: this
cost model (compression/hawq/finn_cost_model.py) only depends on layer
geometry + bit-width, never real weight values, same convention every
other ilp_search.py/folding_ilp.py run in this repo already uses.

Bypasses folding_ilp.py's own --config module system entirely: that
system's hardcoded ENet(...) construction (in finn_stage_costs.py/
finn_block_costs.py) never threads through dsc_no_projection or use_prelu,
both load-bearing for this architecture (dsc_no_projection=1 replaces most
context-stage bottlenecks with DSCNoProjectionBottleneck; use_prelu=0 is
S8.2's own defining ReLU trait) -- using that path would silently cost the
WRONG architecture. Builds the real model directly and calls
folding_ilp.py's own solve_folding() with explicit LayerGeometry objects
instead.

Usage:
    python compression/folding_8_2_relu_w24_no_reg_d2_projected.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compression" / "hawq"))

from nnunetv2.nets.ENet import ENet  # noqa: E402
from finn_cost_model import LayerGeometry  # noqa: E402
from folding_ilp import XCZU7EV, solve_folding  # noqa: E402

IN_CHANNELS = 1
OUT_CHANNELS = 5
INPUT_HW = (512, 512)
WEIGHT_BITS = 8
ACT_BITS = 8

CONFIG = dict(
    channels=(4, 12, 24, 12, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1),
    decoder_type="upsample_conv", use_dilated=True, use_asymmetric=False, use_strided=True,
    use_dsc=False, context_pattern="dense_dilation_d2_projected", use_prelu=False,
    dsc_no_projection=True, dsc_no_projection_context_only=False, reg_bookend_dsc=False,
)

STAGE_PREFIXES = [
    ("initial", "initial"),
    ("stage1", "down1"), ("stage1", "regular1"),
    ("context", "down2"), ("context", "stage2"), ("context", "stage3"),
    ("stage4", "up4"), ("stage4", "regular4"),
    ("stage5", "up5"), ("stage5", "regular5"), ("stage5", "final"),
]


def _stage_of(layer_name: str) -> str:
    for stage, prefix in STAGE_PREFIXES:
        if layer_name == prefix or layer_name.startswith(prefix + "."):
            return stage
    return "?"


def dump_layer_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    geometries: list[LayerGeometry] = []

    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)

    def make_hook(name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            groups = getattr(module, "groups", 1)
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=_stage_of(name),
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw, groups=groups,
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, "Conv2d")))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(make_hook(name, "ConvTranspose2d")))
        elif isinstance(module, nn.MaxPool2d):
            handles.append(module.register_forward_hook(make_hook(name, "MaxPool2d")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, *input_hw))
    for h in handles:
        h.remove()
    return geometries


def main() -> None:
    model = ENet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, **CONFIG)
    geometries = dump_layer_geometry(model, INPUT_HW)
    print(f"Traced {len(geometries)} Conv2d/ConvTranspose2d/MaxPool2d layers.")

    result = solve_folding(
        geometries, stage_bits=None, weight_bits=WEIGHT_BITS, act_bits=ACT_BITS,
        lut_weight=1.0, bram_weight=1.0, hard_lut=1.0, hard_bram=1.0,
    )
    print(f"Status: {result['status']}")
    diag = result["_diagnostics"]
    print(f"LUT: {diag['total_lut_calibrated']:.0f} / {XCZU7EV['LUT']} ({diag['lut_pct_of_budget']:.1f}%)")
    print(f"BRAM_18K: {diag['total_bram18k_calibrated']:.0f} / {XCZU7EV['BRAM_18K']} ({diag['bram_pct_of_budget']:.1f}%)")
    print(f"Cycles: {diag['total_cycles']:.0f} ({diag['total_cycles'] / 100e6 * 1000:.1f} ms @ 100MHz)")
    print(diag["note"])

    out_file = REPO_ROOT / "compression" / "hawq" / "folding_8_2_relu_w24_no_reg_d2_projected_int8_100pct.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
