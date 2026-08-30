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
import importlib
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
# Static default (config_23_1) -- kept as a real import, not just a
# load_config() call, so a library-style `from finn_stage_costs import
# dump_layer_geometry, INPUT_HW` (see folding_ilp.py) still gets working
# STAGE_MODULE_ATTRS/IN_CHANNELS globals immediately at import time, without
# needing to run this file's own main()/load_config() first. Running this
# file directly with --config overrides these via load_config() below,
# same pattern as sensitivity.py.
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CANDIDATE_BITS, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE,
    IN_CHANNELS, OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, STAGE_MODULE_ATTRS,
    STAGE_NAMES, USE_ASYMMETRIC,
)
from finn_cost_model import FOLDING_SERIAL, FOLDING_UNFOLDED, Folding, LayerGeometry, layer_cost  # noqa: E402

INPUT_HW = (512, 512)  # real nnU-Net patch size (see debug.json's configuration_manager.patch_size)


def load_config(config_module: str) -> None:
    """See sensitivity.py's own load_config -- same pattern, injects the
    named config_*.py's constants into this module's globals, overriding
    the static config_23_1 default imported above."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


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


def build_stage_cost_table(geometries: list[LayerGeometry], folding: Folding) -> dict:
    """{stage: {"W{w}_A{a}": {total_lut, total_pe, total_simd_lanes,
    swu_bram18, wm_bram18}}} for every (weight_bits, act_bits) in
    CANDIDATE_BITS x CANDIDATE_BITS, at the given folding config --
    additive-per-layer assumption (each layer's cost computed independently
    at the stage's chosen bits and summed), same assumption hawq/ILP.ipynb's
    own per-layer BOPS/size/latency sums already make."""
    table = {stage: {} for stage in STAGE_NAMES}
    for stage in STAGE_NAMES:
        stage_layers = [g for g in geometries if g.stage == stage]
        for w in CANDIDATE_BITS:
            for a in CANDIDATE_BITS:
                totals = {"total_lut": 0, "total_pe": 0, "total_simd_lanes": 0, "swu_bram18": 0, "wm_bram18": 0}
                for layer in stage_layers:
                    cost = layer_cost(layer, w, a, folding)
                    for k in totals:
                        totals[k] += cost[k]
                table[stage][f"W{w}_A{a}"] = totals
    return table


XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}  # see hardware/finn_estimate_original_enet_unfolded.py's own numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_23_1 or config_21_2.")
    parser.add_argument("--folding", choices=["unfolded", "serial"], default="unfolded",
                         help="'unfolded' (default, matches every existing report in this repo): Q=C_in*K_h*K_w, "
                              "P=C_out, max resource/min latency. 'serial': Q=P=1, min resource/max latency -- "
                              "see finn_cost_model.py's own docstring for why SWU BRAM is identical either way.")
    parser.add_argument("--out-file", type=Path, default=None,
                         help="Defaults to compression/hawq/finn_stage_costs_<config suffix>[_serial].json.")
    args = parser.parse_args()
    if args.config != "config_23_1":
        load_config(args.config)
    folding: Folding = FOLDING_SERIAL if args.folding == "serial" else FOLDING_UNFOLDED
    # config_23_1 keeps the original unsuffixed filename (ilp_search.py's
    # own default --finn-cost-file points there) -- only non-default
    # configs get a name suffix, so this stays backward compatible.
    config_suffix = "" if args.config == "config_23_1" else f"_{args.config.removeprefix('config_')}"
    default_name = f"finn_stage_costs{config_suffix}" + ("_serial" if folding == FOLDING_SERIAL else "")
    out_file = args.out_file or Path(f"compression/hawq/{default_name}.json")

    # OPTIONAL config globals -- see sensitivity.py's build_fp32_model for
    # the full rationale (every existing config_*.py never defines these,
    # so .get() preserves their exact prior behavior).
    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=globals().get("REG_BOOKEND_DSC", False),
    )
    geometries = dump_layer_geometry(model, INPUT_HW)
    print(f"Traced {len(geometries)} Conv2d/ConvTranspose2d/MaxPool2d layers across {len(STAGE_NAMES)} stages. Folding: {folding}")
    for stage in STAGE_NAMES:
        n = sum(1 for g in geometries if g.stage == stage)
        print(f"  {stage}: {n} layers")

    table = build_stage_cost_table(geometries, folding)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {out_file}")

    total_lut = total_bram = 0
    for stage in STAGE_NAMES:
        lo = table[stage][f"W{CANDIDATE_BITS[0]}_A{CANDIDATE_BITS[0]}"]
        hi = table[stage][f"W{CANDIDATE_BITS[-1]}_A{CANDIDATE_BITS[-1]}"]
        print(f"  {stage}: total_lut {lo['total_lut']:.0f} (all-{CANDIDATE_BITS[0]}bit) .. {hi['total_lut']:.0f} (all-{CANDIDATE_BITS[-1]}bit)   "
              f"bram18k {lo['swu_bram18']+lo['wm_bram18']:.0f} .. {hi['swu_bram18']+hi['wm_bram18']:.0f}")
        total_lut += lo["total_lut"]
        total_bram += lo["swu_bram18"] + lo["wm_bram18"]
    print(f"\nCHEAPEST (all-{CANDIDATE_BITS[0]}bit) totals: "
          f"LUT={total_lut:.0f} ({100*total_lut/XCZU7EV['LUT']:.1f}% of {XCZU7EV['LUT']} budget)  "
          f"BRAM_18K={total_bram:.0f} ({100*total_bram/XCZU7EV['BRAM_18K']:.1f}% of {XCZU7EV['BRAM_18K']} budget)")


if __name__ == "__main__":
    main()
