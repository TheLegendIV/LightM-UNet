"""Per-BOTTLENECK-BLOCK FINN hardware cost table -- same closed-form FINN-R
cost formulae as finn_stage_costs.py, evaluated at every (weight_bits,
act_bits) candidate pair, but grouped by INDIVIDUAL ENet bottleneck block
(see block_utils.enumerate_blocks) instead of finn_stage_costs.py's static
5-way stage grouping. Produces a finn_block_costs_<config>.json that
ilp_search.py can consume directly alongside block_sensitivity.py's own
output, to assign every bottleneck its own independent W/A bit-width.

See finn_stage_costs.py's own module docstring for why this is a pure
closed-form estimate (no FINN toolchain/Docker build needed) -- not
repeated here, this file only changes the GROUPING, not the cost formulae
or the "trace geometry once on an untrained FP32 model" approach.

Usage:
    python compression/hawq/finn_block_costs.py \\
        --config config_26_5_w24 \\
        --out-file compression/hawq/finn_block_costs_26_5_w24.json
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
from block_utils import enumerate_blocks, path_to_block_map  # noqa: E402
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CANDIDATE_BITS, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE,
    IN_CHANNELS, OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, USE_ASYMMETRIC,
)
from finn_cost_model import FOLDING_SERIAL, FOLDING_UNFOLDED, Folding, LayerGeometry, layer_cost  # noqa: E402

INPUT_HW = (512, 512)  # real nnU-Net patch size (see debug.json's configuration_manager.patch_size)


def load_config(config_module: str) -> None:
    """Same pattern as finn_stage_costs.py's own loader -- injects the
    named config_*.py's constants into this module's globals."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


def dump_block_layer_geometry(model: nn.Module, input_hw: tuple[int, int]) -> list[LayerGeometry]:
    """Same forward-hook mechanism as finn_stage_costs.py's own
    dump_layer_geometry, but tags each layer with its owning BLOCK (via
    block_utils.path_to_block_map's exact full-path lookup) instead of one
    of 5 static stage names."""
    blocks = enumerate_blocks(model)
    path_to_block = path_to_block_map(blocks)
    geometries: list[LayerGeometry] = []

    def make_hook(name: str, block_name: str, op_type: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(output, tuple):  # MaxPool2d(return_indices=True) -> (values, indices)
                output = output[0]
            kh, kw = _pair(module.kernel_size)
            sh, sw = _pair(module.stride)
            dh, dw = _pair(getattr(module, "dilation", 1))
            geometries.append(LayerGeometry(
                op_type=op_type, name=name, stage=block_name,
                cin=x.shape[1], hin=x.shape[2], win=x.shape[3],
                cout=output.shape[1], hout=output.shape[2], wout=output.shape[3],
                kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw,
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        block_name = path_to_block.get(name)
        if block_name is None:
            continue
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "Conv2d")))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "ConvTranspose2d")))
        elif isinstance(module, nn.MaxPool2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "MaxPool2d")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, *input_hw))
    for h in handles:
        h.remove()
    return geometries, list(blocks.keys())


def build_block_cost_table(geometries: list[LayerGeometry], block_names: list[str], folding: Folding) -> dict:
    """Same {block: {"W{w}_A{a}": {...}}} shape finn_stage_costs.py's own
    build_stage_cost_table produces, just keyed by block instead of stage."""
    table = {block: {} for block in block_names}
    for block in block_names:
        block_layers = [g for g in geometries if g.stage == block]
        for w in CANDIDATE_BITS:
            for a in CANDIDATE_BITS:
                totals = {"total_lut": 0, "total_pe": 0, "total_simd_lanes": 0, "swu_bram18": 0, "wm_bram18": 0}
                for layer in block_layers:
                    cost = layer_cost(layer, w, a, folding)
                    for k in totals:
                        totals[k] += cost[k]
                table[block][f"W{w}_A{a}"] = totals
    return table


XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_23_1 or config_26_5_w24.")
    parser.add_argument("--folding", choices=["unfolded", "serial"], default="unfolded")
    parser.add_argument("--out-file", type=Path, default=None,
                         help="Defaults to compression/hawq/finn_block_costs_<config suffix>[_serial].json.")
    args = parser.parse_args()
    if args.config != "config_23_1":
        load_config(args.config)
    folding: Folding = FOLDING_SERIAL if args.folding == "serial" else FOLDING_UNFOLDED
    config_suffix = "" if args.config == "config_23_1" else f"_{args.config.removeprefix('config_')}"
    default_name = f"finn_block_costs{config_suffix}" + ("_serial" if folding == FOLDING_SERIAL else "")
    out_file = args.out_file or Path(f"compression/hawq/{default_name}.json")

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
    print(f"Traced {len(geometries)} Conv2d/ConvTranspose2d/MaxPool2d layers across {len(block_names)} blocks. Folding: {folding}")
    for block in block_names:
        n = sum(1 for g in geometries if g.stage == block)
        print(f"  {block}: {n} layers")

    table = build_block_cost_table(geometries, block_names, folding)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {out_file}")

    total_lut = total_bram = 0
    for block in block_names:
        lo = table[block][f"W{CANDIDATE_BITS[0]}_A{CANDIDATE_BITS[0]}"]
        hi = table[block][f"W{CANDIDATE_BITS[-1]}_A{CANDIDATE_BITS[-1]}"]
        print(f"  {block}: total_lut {lo['total_lut']:.0f} (all-{CANDIDATE_BITS[0]}bit) .. {hi['total_lut']:.0f} (all-{CANDIDATE_BITS[-1]}bit)   "
              f"bram18k {lo['swu_bram18']+lo['wm_bram18']:.0f} .. {hi['swu_bram18']+hi['wm_bram18']:.0f}")
        total_lut += lo["total_lut"]
        total_bram += lo["swu_bram18"] + lo["wm_bram18"]
    print(f"\nCHEAPEST (all-{CANDIDATE_BITS[0]}bit) totals: "
          f"LUT={total_lut:.0f} ({100*total_lut/XCZU7EV['LUT']:.1f}% of {XCZU7EV['LUT']} budget)  "
          f"BRAM_18K={total_bram:.0f} ({100*total_bram/XCZU7EV['BRAM_18K']:.1f}% of {XCZU7EV['BRAM_18K']} budget)")


if __name__ == "__main__":
    main()
