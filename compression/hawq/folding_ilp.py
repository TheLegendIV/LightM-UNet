"""Per-LAYER folding ILP -- for a given architecture (--config) and,
optionally, a per-stage bit-width assignment already chosen by ilp_search.py
(--stage-bits-file, e.g. compression/hawq/stage_bits_26_5_w24.json), answers
"what PE/SIMD (folding) per layer minimizes latency". Folding is a separate
design axis from the per-stage bit-width search in ilp_search.py -- this
script takes bit-width as GIVEN (either uniform --weight-bits/--act-bits, the
original W8A8 default, or the real per-stage mix from a stage_bits_*.json)
and only searches PE/SIMD.

Folding recap (see finn_cost_model.py's own docstring for the full
derivation): PE = output channels computed in parallel per cycle, SIMD =
reduction elements (C_in*K_h*K_w) processed in parallel per PE per cycle.
FOLDING_UNFOLDED (PE=C_out, SIMD=C_in*K_h*K_w) = one cycle per output
pixel, maximum resource, minimum latency. FOLDING_SERIAL (PE=SIMD=1) = one
MAC per cycle, minimum resource, maximum latency. This ILP searches the
FULL space between those two extremes, per layer independently (each
layer's own C_out/C_in*K_h*K_w have their own valid divisors -- FINN's real
folding constraint: PE must evenly divide C_out, SIMD must evenly divide
C_in*K_h*K_w, no ragged folding).

Variables: one binary x[layer, pe, simd] per (layer, valid PE, valid SIMD)
triple, exactly one chosen per layer (sum == 1).

Objective -- LUT/BRAM are a PENALTY, NOT a hard constraint (changed from
this file's original all-hard-constraint formulation): same rationale
ilp_search.py's own docstring already gives for treating BRAM as a penalty
there (the fully-unfolded/no-folding cost model is frequently well over
budget, so a hard `<=` constraint risks outright infeasibility -- see this
script's own earlier INFEASIBLE-at-W8A8 finding), PLUS this repo's own
observation that real Vivado synthesis typically comes in well under this
closed-form analytical LUT/BRAM estimate (it doesn't model FINN's actual
resource sharing/optimization), so treating the estimate as a soft steering
signal rather than a hard wall is the more honest choice, not just a
convenience. cycles/LUT/BRAM are on wildly different scales (cycles range
from ~1e2 to ~1e8 per candidate depending on folding; LUT ~1e2-1e5; BRAM
~1e0-1e3), so each is independently min-max normalized to [0,1] over the
FULL (layer, pe, simd) candidate space before combining -- same convention
ilp_search.py's own `_normalize` uses for sensitivity/BRAM, just extended to
a 3rd term here:
    minimize sum_{l,pe,simd} x[l,pe,simd] * (cycles_norm[l,pe,simd]
                                              + lut_weight * lut_norm[l,pe,simd]
                                              + bram_weight * bram_norm[l,pe,simd])
Constraint (the only hard one): sum_{pe,simd} x[layer,pe,simd] == 1 per layer
-- exactly one folding choice per layer. No LUT/BRAM `<=` constraint at all.

Chosen LUT/BRAM/cycles totals are still computed and reported after solving
(both on stdout and in the output JSON's "_diagnostics") -- informational,
not a pass/fail gate, per the module's own reasoning above.

Usage (per-STAGE bits, the original 5-group W8A8/mixed case):
    python compression/hawq/folding_ilp.py --out-file compression/hawq/folding_23_1_w8a8.json
    python compression/hawq/folding_ilp.py --config config_26_5_w24 \\
        --stage-bits-file compression/hawq/stage_bits_26_5_w24.json \\
        --lut-weight 1.0 --bram-weight 1.0 \\
        --out-file compression/hawq/folding_26_5_w24.json

Usage (per-BLOCK bits -- ilp_search.py's finer-grained output, one
independent choice per ENet bottleneck instead of one shared choice per
5-way stage group -- needs --granularity block so each layer's geometry
gets tagged by its owning BLOCK, matching --stage-bits-file's own per-block
keys, not the 5 stage names):
    python compression/hawq/folding_ilp.py --config config_26_5_w24 \\
        --granularity block \\
        --stage-bits-file compression/hawq/block_bits_26_5_w24.json \\
        --lut-weight 1.0 --bram-weight 1.0 \\
        --out-file compression/hawq/folding_block_26_5_w24.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pulp
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, IN_CHANNELS,
    OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, USE_ASYMMETRIC,
)
from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from finn_cost_model import LayerGeometry, divisors, layer_cost_pe_simd, max_pe, max_simd  # noqa: E402
import finn_stage_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}


def load_config(config_module: str) -> None:
    """Same pattern as sensitivity.py/finn_stage_costs.py's own loader --
    injects the named config_*.py's constants into this module's globals,
    overriding the static config_23_1 default imported above."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def candidate_folds(layer: LayerGeometry) -> list[tuple[int, int]]:
    """Every valid (PE, SIMD) pair for this layer -- MaxPool2d has neither
    (no MVAU, no weights), so it gets the single sentinel (1, 1), which
    layer_cost_pe_simd ignores for that op_type anyway (its cost/cycles
    don't depend on pe/simd/ram_style at all -- see maxpool_cost). Conv2d/
    ConvTranspose2d get BOTH ram_style options per (PE, SIMD) -- the ILP
    picks whichever (block=BRAM vs ultra=URAM) fits/minimizes cycles per
    layer, since real FINN's LUT/cycle cost is identical either way (only
    the BRAM_18K vs URAM resource ledger differs, see finn_cost_model.py's
    conv_cost_pe_simd docstring)."""
    if layer.op_type == "MaxPool2d":
        return [(1, 1, "block")]
    return [
        (pe, simd, ram_style)
        for pe in divisors(max_pe(layer)) for simd in divisors(max_simd(layer)) for ram_style in RAM_STYLES
    ]


def layer_bits(layer: LayerGeometry, stage_bits: dict | None, weight_bits: int, act_bits: int) -> tuple[int, int]:
    """(weight_bits, act_bits) for this layer -- per-stage from stage_bits
    (an ilp_search.py output: {"stage_weight_bits": {...}, "stage_act_bits":
    {...}}, one entry per STAGE_NAMES) if given, else the uniform CLI
    --weight-bits/--act-bits fallback (this file's original W8A8-everywhere
    behavior)."""
    if stage_bits is None:
        return weight_bits, act_bits
    return stage_bits["stage_weight_bits"][layer.stage], stage_bits["stage_act_bits"][layer.stage]


def _normalize(values: dict) -> dict:
    """Min-max scale to [0,1] -- same helper ilp_search.py's own _normalize
    implements, duplicated here rather than imported (kept self-contained,
    same "one source of truth per CONFIG, not per tiny helper" scope every
    other compression/hawq/ script already follows)."""
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def solve_folding(
    geometries: list[LayerGeometry], stage_bits: dict | None, weight_bits: int, act_bits: int,
    lut_weight: float, bram_weight: float,
) -> dict:
    x = {}
    layer_costs: dict[str, dict[tuple[int, int], dict]] = {}
    raw_cycles: dict[tuple[str, int, int], float] = {}
    raw_lut: dict[tuple[str, int, int], float] = {}
    raw_bram: dict[tuple[str, int, int], float] = {}

    for layer in geometries:
        w_bits, a_bits = layer_bits(layer, stage_bits, weight_bits, act_bits)
        folds = candidate_folds(layer)
        costs = {(pe, simd): layer_cost_pe_simd(layer, w_bits, a_bits, pe, simd) for pe, simd in folds}
        layer_costs[layer.name] = costs
        for pe, simd in folds:
            key = (layer.name, pe, simd)
            raw_cycles[key] = costs[(pe, simd)]["cycles"]
            raw_lut[key] = costs[(pe, simd)]["total_lut"]
            raw_bram[key] = costs[(pe, simd)]["swu_bram18"] + costs[(pe, simd)]["wm_bram18"]
            x[key] = pulp.LpVariable(f"x_{layer.name}_{pe}_{simd}", cat=pulp.LpBinary)

    cycles_norm = _normalize(raw_cycles)
    lut_norm = _normalize(raw_lut)
    bram_norm = _normalize(raw_bram)

    prob = pulp.LpProblem("HAWQ_folding", pulp.LpMinimize)
    for layer in geometries:
        folds = candidate_folds(layer)
        prob += pulp.lpSum(x[(layer.name, pe, simd)] for pe, simd in folds) == 1, f"one_fold_{layer.name}"

    # LUT/BRAM are a PENALTY here, not a hard `<=budget` constraint -- see
    # module docstring for why (fully-unfolded LUT/BRAM is often over budget
    # regardless of folding choice, and real synthesis tends to land well
    # under this closed-form estimate anyway).
    prob += pulp.lpSum(
        x[key] * (cycles_norm[key] + lut_weight * lut_norm[key] + bram_weight * bram_norm[key])
        for key in x
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_name = pulp.LpStatus[status]
    result_per_layer = {}
    if status_name == "Optimal":
        for layer in geometries:
            for pe, simd in candidate_folds(layer):
                if pulp.value(x[(layer.name, pe, simd)]) > 0.5:
                    w_bits, a_bits = layer_bits(layer, stage_bits, weight_bits, act_bits)
                    result_per_layer[layer.name] = {
                        "stage": layer.stage, "pe": pe, "simd": simd,
                        "weight_bits": w_bits, "act_bits": a_bits,
                        **layer_costs[layer.name][(pe, simd)],
                    }
                    break

    total_lut = sum(v["total_lut"] for v in result_per_layer.values())
    total_bram = sum(v["swu_bram18"] + v["wm_bram18"] for v in result_per_layer.values())
    total_cycles = sum(v["cycles"] for v in result_per_layer.values())
    return {
        "status": status_name,
        "per_layer": result_per_layer,
        "_diagnostics": {
            "total_lut": total_lut, "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k": total_bram, "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "total_cycles": total_cycles,
            "note": "LUT/BRAM are informational only here (penalty in the objective via "
                    "--lut-weight/--bram-weight, NOT a hard constraint) -- see this file's own "
                    "module docstring for why. Over 100% does not mean the search failed.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_23_1 or config_26_5_w24.")
    parser.add_argument("--stage-bits-file", type=Path, default=None,
                         help="ilp_search.py output ({'stage_weight_bits': {...}, 'stage_act_bits': {...}}) "
                              "for a real per-stage bit assignment -- e.g. compression/hawq/stage_bits_26_5_w24.json. "
                              "Omit to fall back to uniform --weight-bits/--act-bits everywhere (this file's "
                              "original W8A8 behavior).")
    parser.add_argument("--weight-bits", type=int, default=8, help="Uniform fallback when --stage-bits-file is not given.")
    parser.add_argument("--act-bits", type=int, default=8, help="Uniform fallback when --stage-bits-file is not given.")
    parser.add_argument("--granularity", choices=["stage", "block"], default="stage",
                         help="'stage' (default) tags each layer with one of ENet's 5 stage groups (initial/"
                              "stage1/context/stage4/stage5), matching sensitivity.py/finn_stage_costs.py/"
                              "ilp_search.py's stage-level output (e.g. stage_bits_23_1.json). 'block' tags "
                              "each layer with its owning individual bottleneck (e.g. 'stage2.3'), matching "
                              "block_sensitivity.py/finn_block_costs.py/ilp_search.py's finer-grained output "
                              "(e.g. block_bits_26_5_w24.json). Must match whatever --stage-bits-file's own "
                              "keys actually are -- a mismatch means every lookup in layer_bits() misses.")
    parser.add_argument("--lut-weight", type=float, default=1.0,
                         help="Weight on normalized total LUT in the objective, added to normalized cycles "
                              "(both in [0,1] per-candidate) -- 0 disables the LUT penalty entirely (pure "
                              "latency minimization).")
    parser.add_argument("--bram-weight", type=float, default=1.0, help="Same as --lut-weight, for BRAM_18K.")
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/folding_23_1_w8a8.json"))
    args = parser.parse_args()
    if args.out_file is None:
        suffix = args.config.removeprefix("config_")
        args.out_file = Path(f"compression/hawq/folding_{suffix}_w8a8.json")

    if args.config != "config_23_1":
        load_config(args.config)

    stage_bits = None
    if args.stage_bits_file is not None:
        with open(args.stage_bits_file) as f:
            stage_bits = json.load(f)

    model = ENet(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=cfg.CHANNELS,
        bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE, decoder_type=cfg.DECODER_TYPE,
        use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN,
        separable_dilated=cfg.SEPARABLE_DILATED, use_prelu=True, prelu_variant=cfg.PRELU_VARIANT,
    )
    if args.granularity == "block":
        geometries, _block_names = dump_block_layer_geometry(model, INPUT_HW)
    else:
        geometries = dump_layer_geometry(model, INPUT_HW)
    n_candidates = sum(len(candidate_folds(g)) for g in geometries)
    print(f"Traced {len(geometries)} layers ({args.granularity} granularity), {n_candidates} total "
          f"(layer, PE, SIMD) candidates. "
          f"Bits: {'per-' + args.granularity + ' from ' + str(args.stage_bits_file) if stage_bits else f'uniform W{args.weight_bits}A{args.act_bits}'}. Solving...")

    result = solve_folding(geometries, stage_bits, args.weight_bits, args.act_bits, args.lut_weight, args.bram_weight)
    print(f"ILP status: {result['status']}")
    if result["status"] == "Optimal":
        diag = result["_diagnostics"]
        print(f"Total LUT: {diag['total_lut']:.0f} ({diag['lut_pct_of_budget']:.1f}% of {XCZU7EV['LUT']} budget)")
        print(f"Total BRAM_18K: {diag['total_bram18k']:.0f} ({diag['bram_pct_of_budget']:.1f}% of {XCZU7EV['BRAM_18K']} budget)")
        print(f"Total cycles (sum, ~= per-image latency): {diag['total_cycles']:.0f}")
        print("(LUT/BRAM are informational -- see module docstring, no hard budget was enforced.)")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
