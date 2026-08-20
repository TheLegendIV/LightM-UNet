"""Per-stage FINN hardware cost table for nnUNetTrainerENet_23_1_s19_
warmstart_4c, evaluated at every (weight_bits, act_bits) candidate pair --
the budget input compression/hawq/ilp_search.py's ILP constrains against.

Uses compression/hawq/finn_cost_model.py's closed-form FINN-R cost formulae
(the same ones already used to produce this repo's real FINN estimate
reports) rather than actually building/exporting a QuantENet23_1 model and
running it through FINN's own Docker toolchain at every candidate bit-width:
layer GEOMETRY (channel counts, feature-map sizes, kernel/stride/dilation)
is entirely determined by the architecture, independent of which bit-width
is later chosen, so it only needs to be traced ONCE (via forward hooks on a
freshly-constructed, untrained FP32 ENet -- weight VALUES don't affect
shapes) and then the cost formula can be evaluated in closed form at every
(W, A) pair directly -- no FINN toolchain/Docker container needed for this
step at all.

Usage:
    python compression/hawq/finn_stage_costs.py \\
        --out-file compression/hawq/finn_stage_costs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CANDIDATE_BITS, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE,
    IN_CHANNELS, OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, STAGE_MODULE_ATTRS,
    STAGE_NAMES, USE_ASYMMETRIC,
)
from finn_cost_model import LayerGeometry, layer_cost  # noqa: E402

INPUT_HW = (512, 512)  # real nnU-Net patch size (see debug.json's configuration_manager.patch_size)


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


def dump_layer_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    """Forward-hooks every Conv2d/ConvTranspose2d/MaxPool2d, reading
    channel/spatial dims from the actual input/output TENSORS (robust
    across all three module types, which don't share one consistent
    "in_channels" attribute convention) and kernel/stride/dilation from the
    module's own attributes (all three DO expose .kernel_size/.stride/
    .dilation consistently)."""
    attr_to_stage = {attr: stage for stage, attrs in STAGE_MODULE_ATTRS.items() for attr in attrs}
    geometries: list[LayerGeometry] = []

    def make_hook(name: str, stage: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):  # MaxPool2d(return_indices=True) -> (values, indices)
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=stage,
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw,
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        top_attr = name.split(".", 1)[0]
        stage = attr_to_stage.get(top_attr)
        if stage is None:
            continue
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, stage, "Conv2d")))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(make_hook(name, stage, "ConvTranspose2d")))
        elif isinstance(module, nn.MaxPool2d):
            handles.append(module.register_forward_hook(make_hook(name, stage, "MaxPool2d")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, *input_hw))
    for h in handles:
        h.remove()
    return geometries


def build_stage_cost_table(geometries: list[LayerGeometry]) -> dict:
    """{stage: {"W{w}_A{a}": {total_lut, total_pe, total_simd_lanes,
    swu_bram18, wm_bram18}}} for every (weight_bits, act_bits) in
    CANDIDATE_BITS x CANDIDATE_BITS -- additive-per-layer assumption (each
    layer's cost computed independently at the stage's chosen bits and
    summed), same assumption hawq/ILP.ipynb's own per-layer BOPS/size/
    latency sums already make."""
    table = {stage: {} for stage in STAGE_NAMES}
    for stage in STAGE_NAMES:
        stage_layers = [g for g in geometries if g.stage == stage]
        for w in CANDIDATE_BITS:
            for a in CANDIDATE_BITS:
                totals = {"total_lut": 0, "total_pe": 0, "total_simd_lanes": 0, "swu_bram18": 0, "wm_bram18": 0}
                for layer in stage_layers:
                    cost = layer_cost(layer, w, a)
                    for k in totals:
                        totals[k] += cost[k]
                table[stage][f"W{w}_A{a}"] = totals
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/finn_stage_costs.json"))
    args = parser.parse_args()

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    geometries = dump_layer_geometry(model, INPUT_HW)
    print(f"Traced {len(geometries)} Conv2d/ConvTranspose2d/MaxPool2d layers across {len(STAGE_NAMES)} stages.")
    for stage in STAGE_NAMES:
        n = sum(1 for g in geometries if g.stage == stage)
        print(f"  {stage}: {n} layers")

    table = build_stage_cost_table(geometries)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {args.out_file}")
    for stage in STAGE_NAMES:
        lo = table[stage][f"W{CANDIDATE_BITS[0]}_A{CANDIDATE_BITS[0]}"]["total_lut"]
        hi = table[stage][f"W{CANDIDATE_BITS[-1]}_A{CANDIDATE_BITS[-1]}"]["total_lut"]
        print(f"  {stage}: total_lut ranges {lo:.0f} (all-{CANDIDATE_BITS[0]}bit) .. {hi:.0f} (all-{CANDIDATE_BITS[-1]}bit)")


if __name__ == "__main__":
    main()
