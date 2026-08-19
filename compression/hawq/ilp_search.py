"""Per-stage bit-width ILP search for nnUNetTrainerENet_23_1_s19_warmstart_4c
-- modeled on hawq/ILP.ipynb's own formulation (PuLP), reworked for a 3-way
per-stage choice (weights/activations independently in {2,4,8}, one-hot
binary variables) instead of HAWQ's original 2-way LpVariable(1,2), and for
a FINN LUT-cost budget instead of HAWQ's generic params x bits/BOPs proxy.

For each stage s and candidate bit b, one binary x[s,b] with
sum_b x[s,b] == 1 (exactly one bit-width per stage). Weights and
activations are solved as two INDEPENDENT small MIPs (x_w against
sensitivity_w/weight-side LUT cost, x_a against sensitivity_a/activation-
side LUT cost) -- matches how compression/hawq/sensitivity.py already keeps
weight and activation sensitivity separate, and avoids the joint W*A LUT
coupling finn_cost_model.py's mvu_lut term has (a fully joint ILP would
need one variable per (stage, w, a) triple instead of two independent
(stage, bit) ones -- deliberately not done here, see finn_stage_costs.json's
own W{w}_A{a} keys if a joint formulation is ever wanted later).

Objective (same sign convention as ILP.ipynb's own `sum((x_i-1)*diff_i)` --
adapted to a one-hot sum instead of a 2-way offset):
    minimize sum_s sum_b x[s,b] * sensitivity[s][b]
Budget constraint (same "interpolate between all-lowest-bit and
all-highest-bit total cost by a fraction" convention ILP.ipynb's own
model_size_limit/bops_limit/latency_limit use):
    sum_s sum_b x[s,b] * finn_lut_cost[s][b] <= budget

Usage:
    python compression/hawq/ilp_search.py \\
        --sensitivity-file compression/hawq/sensitivity_23_1.json \\
        --finn-cost-file compression/hawq/finn_stage_costs.json \\
        --weight-budget-fraction 0.5 --act-budget-fraction 0.5 \\
        --out-file compression/hawq/stage_bits_23_1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import CANDIDATE_BITS, STAGE_NAMES  # noqa: E402


def _finn_lut_cost(finn_costs: dict, stage: str, weight_bits: int, act_bits: int) -> float:
    return finn_costs[stage][f"W{weight_bits}_A{act_bits}"]["total_lut"]


def stage_costs_for_axis(finn_costs: dict, stage: str, bit: int, axis: str, other_bits: dict[str, int]) -> float:
    """LUT cost of stage `s` at candidate `bit` on ONE axis (weight or
    activation), holding the OTHER axis fixed at `other_bits[s]` -- needed
    because finn_cost_model.py's mvu_lut term depends on W*A jointly, so a
    weight-only ILP still needs a concrete activation bit-width (and vice
    versa) to look up a real LUT number. `other_bits` should be the
    OTHER axis's already-decided (or a reasonable default, e.g. 8) per-stage
    assignment -- see solve_stage_bits's two-pass call order below."""
    if axis == "weight":
        return _finn_lut_cost(finn_costs, stage, bit, other_bits[stage])
    return _finn_lut_cost(finn_costs, stage, other_bits[stage], bit)


def solve_axis(
    sensitivity: dict, finn_costs: dict, axis: str, other_bits: dict[str, int], budget_fraction: float,
) -> dict[str, int]:
    """One MIP: pick a bit-width per stage on `axis` ('weight' or 'act'),
    minimizing total sensitivity, subject to a LUT budget interpolated
    between the all-lowest-bit and all-highest-bit total cost by
    `budget_fraction` (0 = cheapest/most-quantized-everywhere, 1 =
    all-highest-bit)."""
    sens_key = "sensitivity_w" if axis == "weight" else "sensitivity_a"
    lo_bit, hi_bit = CANDIDATE_BITS[0], CANDIDATE_BITS[-1]

    lo_total = sum(stage_costs_for_axis(finn_costs, s, lo_bit, axis, other_bits) for s in STAGE_NAMES)
    hi_total = sum(stage_costs_for_axis(finn_costs, s, hi_bit, axis, other_bits) for s in STAGE_NAMES)
    budget = lo_total + (hi_total - lo_total) * budget_fraction

    prob = pulp.LpProblem(f"HAWQ_stage_bits_{axis}", pulp.LpMinimize)
    x = {
        (s, b): pulp.LpVariable(f"x_{axis}_{s}_{b}", cat=pulp.LpBinary)
        for s in STAGE_NAMES for b in CANDIDATE_BITS
    }
    for s in STAGE_NAMES:
        prob += pulp.lpSum(x[(s, b)] for b in CANDIDATE_BITS) == 1, f"one_bit_per_stage_{axis}_{s}"

    prob += pulp.lpSum(
        x[(s, b)] * sensitivity[s][sens_key][str(b)] for s in STAGE_NAMES for b in CANDIDATE_BITS
    )
    prob += pulp.lpSum(
        x[(s, b)] * stage_costs_for_axis(finn_costs, s, b, axis, other_bits)
        for s in STAGE_NAMES for b in CANDIDATE_BITS
    ) <= budget, "lut_budget"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"{axis} ILP did not solve to optimality (status={pulp.LpStatus[status]}) -- "
            f"budget_fraction={budget_fraction} may be infeasible (budget={budget:.0f} vs "
            f"lo_total={lo_total:.0f}); try a higher --{axis}-budget-fraction."
        )
    result = {}
    for s in STAGE_NAMES:
        chosen = [b for b in CANDIDATE_BITS if pulp.value(x[(s, b)]) > 0.5]
        assert len(chosen) == 1, f"stage {s} ({axis}): expected exactly one bit chosen, got {chosen}"
        result[s] = chosen[0]
    return result, budget, lo_total, hi_total


def solve_stage_bits(
    sensitivity: dict, finn_costs: dict, weight_budget_fraction: float, act_budget_fraction: float,
) -> dict:
    """Two-pass: solve weights first holding activations at the highest
    candidate bit (a conservative "don't let an unsolved activation choice
    bias the weight LUT lookup toward an artificially cheap number"
    default), then solve activations holding weights at the just-solved
    result. Not a true joint optimum (see module docstring's note on
    finn_cost_model.py's W*A coupling), but keeps two simple independent
    MIPs instead of one combined (stage, w, a) triple-indexed ILP."""
    default_act = {s: CANDIDATE_BITS[-1] for s in STAGE_NAMES}
    stage_weight_bits, w_budget, w_lo, w_hi = solve_axis(sensitivity, finn_costs, "weight", default_act, weight_budget_fraction)
    stage_act_bits, a_budget, a_lo, a_hi = solve_axis(sensitivity, finn_costs, "act", stage_weight_bits, act_budget_fraction)
    return {
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "_diagnostics": {
            "weight_budget_lut": w_budget, "weight_lo_total_lut": w_lo, "weight_hi_total_lut": w_hi,
            "act_budget_lut": a_budget, "act_lo_total_lut": a_lo, "act_hi_total_lut": a_hi,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensitivity-file", type=Path, default=Path("compression/hawq/sensitivity_23_1.json"))
    parser.add_argument("--finn-cost-file", type=Path, default=Path("compression/hawq/finn_stage_costs.json"))
    parser.add_argument("--weight-budget-fraction", type=float, default=0.5,
                         help="0 = cheapest (all-lowest-bit) LUT budget, 1 = all-highest-bit budget.")
    parser.add_argument("--act-budget-fraction", type=float, default=0.5)
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/stage_bits_23_1.json"))
    args = parser.parse_args()

    with open(args.sensitivity_file) as f:
        sensitivity = json.load(f)
    with open(args.finn_cost_file) as f:
        finn_costs = json.load(f)

    result = solve_stage_bits(sensitivity, finn_costs, args.weight_budget_fraction, args.act_budget_fraction)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"stage_weight_bits: {result['stage_weight_bits']}")
    print(f"stage_act_bits:    {result['stage_act_bits']}")


if __name__ == "__main__":
    main()
