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
adapted to a one-hot sum instead of a 2-way offset), NOW a weighted sum of
sensitivity AND BRAM cost, both min-max normalized to [0,1] over the
axis's own (stage, bit) candidates so a single --bram-weight is meaningful
across two differently-scaled quantities (sensitivity ~1e-6..1, BRAM
~1e1..1e4):
    minimize sum_s sum_b x[s,b] * (sensitivity_norm[s][b]
                                    + bram_weight * bram_norm[s][b])
Budget constraint (unchanged, same "interpolate between all-lowest-bit and
all-highest-bit total cost by a fraction" convention ILP.ipynb's own
model_size_limit/bops_limit/latency_limit use) -- still LUT-only, still a
HARD constraint:
    sum_s sum_b x[s,b] * finn_lut_cost[s][b] <= lut_budget

BRAM is a PENALTY in the objective, not a second hard constraint, on
purpose: checked directly (see conversation/this file's own history) that
even the all-2-bit (cheapest possible) assignment is ~4.7x OVER this
architecture's real BRAM_18K budget (624, XCZU7EV) under finn_cost_model.py's
fully-unfolded (M=1, no PE/SIMD folding) convention -- a hard `<= budget`
constraint on BRAM would make the ILP infeasible at ANY bit-width, for ANY
budget_fraction, full stop. Folding (PE/SIMD reduction, trading resource for
latency) is the real lever for actually fitting BRAM/LUT to a chip -- this
repo doesn't build that yet (see hardware/README.md's own open item for the
bigger "original" architecture, same root cause). So: bram_weight steers the
search toward the LOWER end of that still-infeasible range, it does not
(cannot, within this cost model) make the result "fit the FPGA" -- that
still needs a real folding step downstream of whatever bit-width comes out
of this ILP.

Usage (per-STAGE, the original 5-group search):
    python compression/hawq/ilp_search.py \\
        --sensitivity-file compression/hawq/sensitivity_23_1.json \\
        --finn-cost-file compression/hawq/finn_stage_costs.json \\
        --weight-budget-fraction 0.5 --act-budget-fraction 0.5 \\
        --bram-weight 1.0 \\
        --out-file compression/hawq/stage_bits_23_1.json

Usage (per-BLOCK -- one independent choice per ENet bottleneck instead of
one shared choice per 5-way stage group; --sensitivity-file/--finn-cost-file
just need to point at block_sensitivity.py/finn_block_costs.py's own output
instead of sensitivity.py/finn_stage_costs.py's):
    python compression/hawq/ilp_search.py \\
        --sensitivity-file compression/hawq/block_sensitivity_26_5_w24.json \\
        --finn-cost-file compression/hawq/finn_block_costs_26_5_w24.json \\
        --out-file compression/hawq/block_bits_26_5_w24.json

The set of names being solved over (stages or blocks) is NOT hardcoded --
it's read directly from --sensitivity-file's own top-level keys (which must
match --finn-cost-file's own keys 1:1), so this one script transparently
handles either granularity. CANDIDATE_BITS=(2,4,8) IS still a fixed
constant (architecture- and granularity-invariant -- every config_*.py and
both sensitivity/cost scripts agree on the same 3 candidates), no longer
imported from config_23_1 (removes an unnecessary cross-architecture
coupling this file never actually needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pulp

CANDIDATE_BITS = (2, 4, 8)


def _finn_cost(finn_costs: dict, stage: str, weight_bits: int, act_bits: int, metric: str) -> float:
    entry = finn_costs[stage][f"W{weight_bits}_A{act_bits}"]
    if metric == "bram18k":
        return entry["swu_bram18"] + entry["wm_bram18"]
    return entry[metric]


def stage_costs_for_axis(
    finn_costs: dict, stage: str, bit: int, axis: str, other_bits: dict[str, int], metric: str = "total_lut",
) -> float:
    """`metric` cost ("total_lut" or "bram18k" -- swu_bram18+wm_bram18
    combined, the two BRAM terms finn_cost_model.py computes) of stage `s`
    at candidate `bit` on ONE axis (weight or activation), holding the
    OTHER axis fixed at `other_bits[s]` -- needed because finn_cost_
    model.py's mvu_lut term depends on W*A jointly, so a weight-only ILP
    still needs a concrete activation bit-width (and vice versa) to look up
    a real cost number. `other_bits` should be the OTHER axis's already-
    decided (or a reasonable default, e.g. 8) per-stage assignment -- see
    solve_stage_bits's two-pass call order below."""
    if axis == "weight":
        return _finn_cost(finn_costs, stage, bit, other_bits[stage], metric)
    return _finn_cost(finn_costs, stage, other_bits[stage], bit, metric)


def _normalize(values: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Min-max scale to [0,1] -- puts sensitivity and BRAM cost, which
    differ by several orders of magnitude, on a comparable footing so
    --bram-weight is a meaningful single knob rather than requiring the
    caller to know each quantity's raw scale."""
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def solve_axis(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...], axis: str, other_bits: dict[str, int],
    budget_fraction: float, bram_weight: float,
) -> dict[str, int]:
    """One MIP: pick a bit-width per stage on `axis` ('weight' or 'act'),
    minimizing normalized_sensitivity + bram_weight * normalized_bram,
    subject to a LUT budget interpolated between the all-lowest-bit and
    all-highest-bit total LUT cost by `budget_fraction` (0 = cheapest/
    most-quantized-everywhere, 1 = all-highest-bit). LUT stays the only
    HARD constraint -- BRAM is a penalty in the objective, not a second
    constraint, because even the cheapest (all-lowest-bit) BRAM total is
    still way over this architecture's real budget under the fully-unfolded
    cost model (see module docstring) -- a hard BRAM constraint would be
    infeasible regardless of budget_fraction."""
    sens_key = "sensitivity_w" if axis == "weight" else "sensitivity_a"
    lo_bit, hi_bit = CANDIDATE_BITS[0], CANDIDATE_BITS[-1]

    lo_total = sum(stage_costs_for_axis(finn_costs, s, lo_bit, axis, other_bits) for s in stage_names)
    hi_total = sum(stage_costs_for_axis(finn_costs, s, hi_bit, axis, other_bits) for s in stage_names)
    budget = lo_total + (hi_total - lo_total) * budget_fraction

    raw_sensitivity = {
        (s, b): sensitivity[s][sens_key][str(b)] for s in stage_names for b in CANDIDATE_BITS
    }
    raw_bram = {
        (s, b): stage_costs_for_axis(finn_costs, s, b, axis, other_bits, metric="bram18k")
        for s in stage_names for b in CANDIDATE_BITS
    }
    sens_norm = _normalize(raw_sensitivity)
    bram_norm = _normalize(raw_bram)

    prob = pulp.LpProblem(f"HAWQ_stage_bits_{axis}", pulp.LpMinimize)
    x = {
        (s, b): pulp.LpVariable(f"x_{axis}_{s}_{b}", cat=pulp.LpBinary)
        for s in stage_names for b in CANDIDATE_BITS
    }
    for s in stage_names:
        prob += pulp.lpSum(x[(s, b)] for b in CANDIDATE_BITS) == 1, f"one_bit_per_stage_{axis}_{s}"

    prob += pulp.lpSum(
        x[(s, b)] * (sens_norm[(s, b)] + bram_weight * bram_norm[(s, b)])
        for s in stage_names for b in CANDIDATE_BITS
    )
    prob += pulp.lpSum(
        x[(s, b)] * stage_costs_for_axis(finn_costs, s, b, axis, other_bits)
        for s in stage_names for b in CANDIDATE_BITS
    ) <= budget, "lut_budget"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"{axis} ILP did not solve to optimality (status={pulp.LpStatus[status]}) -- "
            f"budget_fraction={budget_fraction} may be infeasible (budget={budget:.0f} vs "
            f"lo_total={lo_total:.0f}); try a higher --{axis}-budget-fraction."
        )
    result = {}
    for s in stage_names:
        chosen = [b for b in CANDIDATE_BITS if pulp.value(x[(s, b)]) > 0.5]
        assert len(chosen) == 1, f"stage {s} ({axis}): expected exactly one bit chosen, got {chosen}"
        result[s] = chosen[0]
    chosen_bram_total = sum(raw_bram[(s, result[s])] for s in stage_names)
    return result, budget, lo_total, hi_total, chosen_bram_total


XCZU7EV_BRAM_18K = 624  # real chip budget, see finn_cost_model.py's own docstring


def solve_stage_bits(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...],
    weight_budget_fraction: float, act_budget_fraction: float, bram_weight: float,
) -> dict:
    """Two-pass: solve weights first holding activations at the highest
    candidate bit (a conservative "don't let an unsolved activation choice
    bias the weight LUT lookup toward an artificially cheap number"
    default), then solve activations holding weights at the just-solved
    result. Not a true joint optimum (see module docstring's note on
    finn_cost_model.py's W*A coupling), but keeps two simple independent
    MIPs instead of one combined (stage, w, a) triple-indexed ILP.

    `stage_names` may be 5 stage-group names or dozens of individual
    bottleneck-block names -- this function has no idea which, and doesn't
    need to (see module docstring)."""
    default_act = {s: CANDIDATE_BITS[-1] for s in stage_names}
    stage_weight_bits, w_budget, w_lo, w_hi, w_bram = solve_axis(
        sensitivity, finn_costs, stage_names, "weight", default_act, weight_budget_fraction, bram_weight,
    )
    stage_act_bits, a_budget, a_lo, a_hi, a_bram = solve_axis(
        sensitivity, finn_costs, stage_names, "act", stage_weight_bits, act_budget_fraction, bram_weight,
    )
    # NOT w_bram + a_bram -- those are partial sums computed under two
    # DIFFERENT "other axis" assumptions (w_bram assumed act=8 everywhere
    # via default_act; a_bram used the just-solved real weight bits), so
    # adding them double-counts under inconsistent bit assumptions. The
    # real combined BRAM at the actual final (w,a) pair per stage:
    total_bram = sum(
        _finn_cost(finn_costs, s, stage_weight_bits[s], stage_act_bits[s], "bram18k") for s in stage_names
    )
    return {
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "_diagnostics": {
            "weight_budget_lut": w_budget, "weight_lo_total_lut": w_lo, "weight_hi_total_lut": w_hi,
            "act_budget_lut": a_budget, "act_lo_total_lut": a_lo, "act_hi_total_lut": a_hi,
            "total_bram18k": total_bram,
            "xczu7ev_bram18k_budget": XCZU7EV_BRAM_18K,
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV_BRAM_18K,
            "note": "BRAM is a penalty in the objective (bram_weight), NOT a hard constraint -- "
                    "even the cheapest bit-width choice is well over budget under the fully-unfolded "
                    "cost model, see ilp_search.py's own module docstring. This number is informational.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensitivity-file", type=Path, default=Path("compression/hawq/sensitivity_23_1.json"))
    parser.add_argument("--finn-cost-file", type=Path, default=Path("compression/hawq/finn_stage_costs.json"))
    parser.add_argument("--weight-budget-fraction", type=float, default=0.5,
                         help="0 = cheapest (all-lowest-bit) LUT budget, 1 = all-highest-bit budget.")
    parser.add_argument("--act-budget-fraction", type=float, default=0.5)
    parser.add_argument("--bram-weight", type=float, default=1.0,
                         help="Weight on normalized BRAM cost in the objective, added to normalized "
                              "sensitivity (both in [0,1]) -- 0 disables BRAM entirely (old behavior), "
                              "1 treats a full BRAM-range shift as equally important as a full "
                              "sensitivity-range shift. See module docstring: this steers toward the "
                              "lower end of BRAM usage, it does NOT guarantee fitting the real budget.")
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/stage_bits_23_1.json"))
    args = parser.parse_args()

    with open(args.sensitivity_file) as f:
        sensitivity = json.load(f)
    with open(args.finn_cost_file) as f:
        finn_costs = json.load(f)

    # The set of names solved over -- 5 stage groups or dozens of individual
    # bottleneck blocks -- comes straight from the sensitivity file's own
    # top-level keys (see module docstring), not a hardcoded import.
    stage_names = tuple(sensitivity.keys())
    missing_in_costs = set(stage_names) - set(finn_costs.keys())
    if missing_in_costs:
        raise ValueError(
            f"--finn-cost-file is missing entries for: {sorted(missing_in_costs)} -- "
            f"--sensitivity-file and --finn-cost-file must cover the exact same names "
            f"(both from sensitivity.py+finn_stage_costs.py, or both from "
            f"block_sensitivity.py+finn_block_costs.py, not a mix of the two granularities)."
        )

    result = solve_stage_bits(
        sensitivity, finn_costs, stage_names, args.weight_budget_fraction, args.act_budget_fraction, args.bram_weight,
    )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"stage_weight_bits: {result['stage_weight_bits']}")
    print(f"stage_act_bits:    {result['stage_act_bits']}")
    diag = result["_diagnostics"]
    print(f"BRAM_18K used: {diag['total_bram18k']:.0f} ({diag['bram_pct_of_budget']:.1f}% of {XCZU7EV_BRAM_18K} budget) -- informational, see module docstring.")


if __name__ == "__main__":
    main()
