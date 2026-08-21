"""FINN INT8 cost estimates for 7 S5.6-derived architecture variants being
considered for HPC training. S5.6 = nnUNetTrainerENet_5_6_separable_dense_dilation
(channels=4,16,32,16,4, bottlenecks_per_stage=4,8,8,2,1, context_pattern=
dense_dilation, separable_dilated=True, dice=0.7985 -- see compression/
results.csv). Uses the same closed-form FINN-R cost formulae as
compression/hawq/finn_cost_model.py/finn_stage_costs.py (no FINN toolchain/
Docker build needed): layer geometry (channels/kernel/stride) is entirely
architecture-determined, so a single untrained FP32 ENet per variant is
enough to trace it.

NOTE on variant 2 (a dilation=1 conv prepended to each dilation cycle, i.e.
d=1,2,4,8,16 repeated twice instead of d=2,4,8,16 repeated twice): approximated
here as context_pattern="dense_dilation" at bottleneck depth 10 instead of 8.
FINN's per-layer cost formulae depend on channels/kernel-shape/stride, NOT on
the dilation rate itself (dilation only changes which input pixels a
fixed-size kernel reads -- see finn_cost_model.py's own docstring), so a
depth-10 dense_dilation stage is numerically IDENTICAL, cost-wise, to a real
d=1,2,4,8,16 x2 pattern with the same depth. Wiring up d=1 as a first-class
context_pattern in ENet.py (so dilation values are actually correct for
training, not just cost estimation) is a separate follow-up once a variant is
chosen for real HPC training.

Variant 6 = config 5 (channels 4,8,24,8,4) + config 3 (no stage3) only --
does NOT include config 2's d=1 lead-in. Variant 7 = variant 6 + config 2
(d=1 lead-in on top), so variant 7's stage2 runs at bottleneck depth 10
where variant 6's stays at the native depth 8. Variant 8 = variant 7 +
config 1's channel halving (/2, floor 4) applied to variant 7's own
channels (4,8,24,8,4 -> 4,4,12,4,4), keeping variant 7's depth-10 stage2
and dropped stage3.

Usage:
    python compression/finn_cost_s5_6_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compression" / "hawq"))

from nnunetv2.nets.ENet import ENet  # noqa: E402
from finn_cost_model import FOLDING_UNFOLDED, LayerGeometry, layer_cost  # noqa: E402

IN_CHANNELS = 1
OUT_CHANNELS = 5
INPUT_HW = (512, 512)
W_BITS = 8
A_BITS = 8
XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}  # see hardware/finn_estimate_original_enet_unfolded.py

COMMON = dict(
    in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, decoder_type="upsample_conv",
    use_dilated=True, use_asymmetric=False, use_strided=True, use_dsc=False,
    context_pattern="dense_dilation", use_prelu=True, prelu_variant="standard",
    separable_dilated=True,
)

VARIANTS = {
    "S5.6 (baseline)": dict(channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1)),
    "V1 U8-style (/2, floor 4)": dict(channels=(4, 8, 16, 8, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1)),
    "V2 d=1 lead-in": dict(channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 10, 10, 2, 1)),
    "V3 no stage3": dict(channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 8, 0, 2, 1)),
    "V4 V2+V3": dict(channels=(4, 16, 32, 16, 4), bottlenecks_per_stage=(4, 10, 0, 2, 1)),
    "V5 channels 4,8,24,8,4": dict(channels=(4, 8, 24, 8, 4), bottlenecks_per_stage=(4, 8, 8, 2, 1)),
    "V6 V5+V3": dict(channels=(4, 8, 24, 8, 4), bottlenecks_per_stage=(4, 8, 0, 2, 1)),
    "V7 V6+V2": dict(channels=(4, 8, 24, 8, 4), bottlenecks_per_stage=(4, 10, 0, 2, 1)),
    "V8 V7+V1": dict(channels=(4, 4, 12, 4, 4), bottlenecks_per_stage=(4, 10, 0, 2, 1)),
}


def dump_layer_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    geometries: list[LayerGeometry] = []

    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)

    def make_hook(name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):  # MaxPool2d(return_indices=True) -> (values, indices)
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage="network",
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw,
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
    rows = []
    for name, cfg in VARIANTS.items():
        model = ENet(**COMMON, **cfg)
        params = sum(p.numel() for p in model.parameters())
        geometries = dump_layer_geometry(model, INPUT_HW)
        total_lut = total_bram = total_pe = total_simd = 0.0
        for g in geometries:
            cost = layer_cost(g, W_BITS, A_BITS, FOLDING_UNFOLDED)
            total_lut += cost["total_lut"]
            total_bram += cost["swu_bram18"] + cost["wm_bram18"]
            total_pe += cost["total_pe"]
            total_simd += cost["total_simd_lanes"]
        rows.append((name, params, total_lut, total_bram, total_pe, total_simd))

    header = f"{'variant':28s} {'params':>9s} {'LUT':>10s} {'LUT %':>8s} {'BRAM18K':>9s} {'BRAM %':>8s} {'PE':>6s} {'SIMD':>7s}"
    print(header)
    print("-" * len(header))
    for name, params, total_lut, total_bram, total_pe, total_simd in rows:
        print(f"{name:28s} {params:9d} {total_lut:10.0f} {100*total_lut/XCZU7EV['LUT']:7.1f}% "
              f"{total_bram:9.0f} {100*total_bram/XCZU7EV['BRAM_18K']:7.1f}% {total_pe:6.0f} {total_simd:7.0f}")


if __name__ == "__main__":
    main()
