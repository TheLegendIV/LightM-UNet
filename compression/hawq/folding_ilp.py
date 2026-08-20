"""Per-LAYER folding ILP for nnUNetTrainerENet_23_1_s19_warmstart_4c at
FIXED W8A8 (uniform 8-bit weights and activations everywhere -- folding is
a completely separate design axis from the per-stage bit-width search in
ilp_search.py, and this script answers a different question: "given a
fixed bit-width, what PE/SIMD (folding) per layer minimizes latency while
fitting the ZCU7's real LUT/BRAM budget").

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
Objective: minimize sum_layer cycles(layer, pe, simd) -- i.e. total
per-image LATENCY (summing every layer's own cycle count, as if the
image passes through the pipeline once, no cross-layer overlap credited).
This is NOT the same as maximizing STEADY-STATE THROUGHPUT (which is
governed by max_layer cycles(layer, pe, simd), the single slowest/
bottleneck layer in a pipelined dataflow design, once many images are
in flight) -- sum is used here because it's directly linear/MIP-friendly
and answers "how long does one image take", the more relevant question
for single-image medical segmentation inference. A throughput-oriented
(minimize max) variant would need one extra continuous variable T with
T >= cycles(layer) for every layer -- straightforward to add later if
wanted, not built here.
Constraints (HARD, unlike ilp_search.py's BRAM-as-penalty choice -- here
BOTH resources get a real shot at being satisfiable since folding, unlike
bit-width alone, has enough range to plausibly reach budget, per the
compression/hawq/finn_stage_costs_serial.json finding that serial+2bit
alone gets BRAM down to 728/624 with W2A2 -- W8A8 will need more folding
headroom to compensate, this ILP finds out whether that's actually
achievable):
    sum_layer LUT(layer, pe, simd) <= 230400  (XCZU7EV LUT budget)
    sum_layer BRAM18K(layer, pe, simd) <= 624  (XCZU7EV BRAM_18K budget)

Usage:
    python compression/hawq/folding_ilp.py --out-file compression/hawq/folding_23_1_w8a8.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pulp
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, IN_CHANNELS,
    OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, USE_ASYMMETRIC,
)
from finn_cost_model import LayerGeometry, divisors, layer_cost_pe_simd, max_pe, max_simd  # noqa: E402
from finn_stage_costs import INPUT_HW, dump_layer_geometry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}
WEIGHT_BITS = 8
ACT_BITS = 8


def candidate_folds(layer: LayerGeometry) -> list[tuple[int, int]]:
    """Every valid (PE, SIMD) pair for this layer -- MaxPool2d has neither
    (no MVAU, no weights), so it gets the single sentinel (1, 1), which
    layer_cost_pe_simd ignores for that op_type anyway (its cost/cycles
    don't depend on pe/simd at all -- see maxpool_cost)."""
    if layer.op_type == "MaxPool2d":
        return [(1, 1)]
    return [(pe, simd) for pe in divisors(max_pe(layer)) for simd in divisors(max_simd(layer))]


def solve_folding(geometries: list[LayerGeometry]) -> dict:
    prob = pulp.LpProblem("HAWQ_folding_w8a8", pulp.LpMinimize)
    x = {}
    layer_costs: dict[str, dict[tuple[int, int], dict]] = {}
    for layer in geometries:
        folds = candidate_folds(layer)
        costs = {(pe, simd): layer_cost_pe_simd(layer, WEIGHT_BITS, ACT_BITS, pe, simd) for pe, simd in folds}
        layer_costs[layer.name] = costs
        for pe, simd in folds:
            x[(layer.name, pe, simd)] = pulp.LpVariable(f"x_{layer.name}_{pe}_{simd}", cat=pulp.LpBinary)
        prob += pulp.lpSum(x[(layer.name, pe, simd)] for pe, simd in folds) == 1, f"one_fold_{layer.name}"

    prob += pulp.lpSum(
        x[(layer.name, pe, simd)] * cost["cycles"]
        for layer in geometries for (pe, simd), cost in layer_costs[layer.name].items()
    )
    prob += pulp.lpSum(
        x[(layer.name, pe, simd)] * cost["total_lut"]
        for layer in geometries for (pe, simd), cost in layer_costs[layer.name].items()
    ) <= XCZU7EV["LUT"], "lut_budget"
    prob += pulp.lpSum(
        x[(layer.name, pe, simd)] * (cost["swu_bram18"] + cost["wm_bram18"])
        for layer in geometries for (pe, simd), cost in layer_costs[layer.name].items()
    ) <= XCZU7EV["BRAM_18K"], "bram_budget"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_name = pulp.LpStatus[status]
    result_per_layer = {}
    if status_name == "Optimal":
        for layer in geometries:
            for pe, simd in candidate_folds(layer):
                if pulp.value(x[(layer.name, pe, simd)]) > 0.5:
                    result_per_layer[layer.name] = {
                        "stage": layer.stage, "pe": pe, "simd": simd,
                        **layer_costs[layer.name][(pe, simd)],
                    }
                    break
    return {"status": status_name, "per_layer": result_per_layer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/folding_23_1_w8a8.json"))
    args = parser.parse_args()

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=True, prelu_variant=PRELU_VARIANT,
    )
    geometries = dump_layer_geometry(model, INPUT_HW)
    n_candidates = sum(len(candidate_folds(g)) for g in geometries)
    print(f"Traced {len(geometries)} layers, {n_candidates} total (layer, PE, SIMD) candidates. Solving...")

    result = solve_folding(geometries)
    print(f"ILP status: {result['status']}")
    if result["status"] != "Optimal":
        print("INFEASIBLE at W8A8 within LUT+BRAM budget -- no folding configuration fits. "
              "See compression/hawq/finn_stage_costs_serial.json for the theoretical floor at W2A2 "
              "(728/624 BRAM, still over) -- W8A8 needs even more headroom, may simply be infeasible.")
    else:
        total_lut = sum(v["total_lut"] for v in result["per_layer"].values())
        total_bram = sum(v["swu_bram18"] + v["wm_bram18"] for v in result["per_layer"].values())
        total_cycles = sum(v["cycles"] for v in result["per_layer"].values())
        print(f"Total LUT: {total_lut:.0f} ({100*total_lut/XCZU7EV['LUT']:.1f}% of budget)")
        print(f"Total BRAM_18K: {total_bram:.0f} ({100*total_bram/XCZU7EV['BRAM_18K']:.1f}% of budget)")
        print(f"Total cycles (sum, ~= per-image latency): {total_cycles:.0f}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
