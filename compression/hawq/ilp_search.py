"""Per-stage/per-block bit-width ILP search -- modeled on hawq/ILP.ipynb's
own formulation (PuLP), reworked for a 3-way per-unit choice (weights/
activations independently in {2,4,8}, one-hot binary variables) instead of
HAWQ's original 2-way LpVariable(1,2), and for FINN LUT/BRAM cost instead of
HAWQ's generic params x bits/BOPs proxy.

For each unit s (a stage-group or an individual bottleneck block -- see
below) and candidate bit b, one binary x[s,b] with sum_b x[s,b] == 1
(exactly one bit-width per unit). By DEFAULT weights and activations are
solved as two INDEPENDENT small MIPs (solve_stage_bits -- x_w against
sensitivity_w/weight-side cost holding act at a default, then x_a against
sensitivity_a/activation-side cost holding weight at the just-solved
result) -- matches how compression/hawq/sensitivity.py already keeps weight
and activation sensitivity separate as two measurements, but only
APPROXIMATES the joint W*A LUT coupling finn_cost_model.py's mvu_lut term
actually has (the weight pass's "hold act at a default" assumption can look
up a cost for a (w, default_a) pair that never gets built once the act pass
picks a different act bit).

Pass --joint to use solve_joint_bits instead: one MIP over all 9 (w,a)
candidate PAIRS per unit (CANDIDATE_BITS x CANDIDATE_BITS), looking up
finn_costs[s][f"W{w}_A{a}"] at the EXACT pair under consideration -- exact
on the LUT/BRAM side, no more two-pass approximation there. Sensitivity
still can't be measured jointly (sensitivity.py/block_sensitivity.py only
ever perturb one axis at a time), so --joint falls back to the standard
additive approximation sensitivity_w[s][w] + sensitivity_a[s][a] -- see
solve_joint_bits's own docstring. --joint is strictly more accurate on
resource cost, at 3x the variable count per unit (9 vs. 3+3) -- CBC solves
either in well under a second even at ~35 blocks, so this isn't a
performance tradeoff, just an accuracy one.

Objective -- LUT AND BRAM are BOTH a PENALTY, NOT a hard constraint (changed
2026-08-25, see below for why LUT's own hard constraint was dropped):
    minimize sum_s sum_b x[s,b] * (sensitivity_norm[s][b]
                                    + lut_weight * lut_norm[s][b]
                                    + bram_weight * bram_norm[s][b])
All three terms independently min-max normalized to [0,1] over the axis's
own (unit, bit) candidates, so --lut-weight/--bram-weight are meaningful
single knobs across quantities that otherwise differ by many orders of
magnitude (sensitivity ~1e-6..1, LUT/BRAM ~1e1..1e6). No hard resource
constraint at all -- see below.

WHY LUT LOST ITS HARD CONSTRAINT (previously the one hard `<= lut_budget`
term, interpolated between the all-lowest-bit and all-highest-bit LUT total
by --weight/act-budget-fraction): calibrated 2026-08-25 against this
repo's one real Vivado synthesis data point (S19 @ uniform W8A8, see
finn_cost_model.py's own LUT_DERATING_FACTOR/BRAM_DERATING_FACTOR module
comment for the full derivation and its real scope limits -- one
architecture, one bit-width, one folding regime, a sum-of-independent-
partitions build). Real LUT usage was ~8.2x THIS model's own raw estimate --
once that correction is applied, LUT becomes exactly as chronically
infeasible as BRAM already was (even the cheapest all-2-bit assignment is
still way over budget), so a hard LUT constraint would make the ILP
infeasible at every budget_fraction, the same reason BRAM was already a
penalty rather than a constraint. Both resources are now treated
consistently: penalties in the objective, using the CALIBRATED (derated)
totals so --lut-weight/--bram-weight steer toward genuinely cheaper designs
rather than an uncorrected estimate that gets the two resources' relative
scale wrong. This also makes this file's own formulation consistent with
compression/hawq/folding_ilp.py's (which already treats both as penalties).
Neither weight "guarantees" fitting the real FPGA -- see finn_cost_model.py's
own derating-factor comment for why that claim can't be made from one
calibration point, and because folding (a separate axis, see folding_ilp.py)
is the real lever for actually reducing resource use, not bit-width alone.

Usage (per-STAGE, the original 5-group search):
    python compression/hawq/ilp_search.py \\
        --sensitivity-file compression/hawq/sensitivity_23_1.json \\
        --finn-cost-file compression/hawq/finn_stage_costs.json \\
        --lut-weight 1.0 --bram-weight 1.0 \\
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
import sys
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finn_cost_model import calibrated_bram18k, calibrated_lut  # noqa: E402

CANDIDATE_BITS = (2, 4, 8)


def _finn_cost(finn_costs: dict, stage: str, weight_bits: int, act_bits: int, metric: str) -> float:
    entry = finn_costs[stage][f"W{weight_bits}_A{act_bits}"]
    if metric == "bram18k":
        return calibrated_bram18k(entry["swu_bram18"] + entry["wm_bram18"])
    if metric == "total_lut":
        return calibrated_lut(entry["total_lut"])
    return entry[metric]


def stage_costs_for_axis(
    finn_costs: dict, stage: str, bit: int, axis: str, other_bits: dict[str, int], metric: str = "total_lut",
) -> float:
    """`metric` cost ("total_lut" or "bram18k" -- swu_bram18+wm_bram18
    combined, both CALIBRATED via finn_cost_model.py's own derating
    factors) of stage `s` at candidate `bit` on ONE axis (weight or
    activation), holding the OTHER axis fixed at `other_bits[s]` -- needed
    because finn_cost_model.py's mvu_lut term depends on W*A jointly, so a
    weight-only ILP still needs a concrete activation bit-width (and vice
    versa) to look up a real cost number. `other_bits` should be the OTHER
    axis's already-decided (or a reasonable default, e.g. 8) per-stage
    assignment -- see solve_stage_bits's two-pass call order below."""
    if axis == "weight":
        return _finn_cost(finn_costs, stage, bit, other_bits[stage], metric)
    return _finn_cost(finn_costs, stage, other_bits[stage], bit, metric)


def _normalize(values: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Min-max scale to [0,1] -- puts sensitivity, LUT, and BRAM cost, which
    differ by several orders of magnitude, on a comparable footing so
    --lut-weight/--bram-weight are meaningful single knobs rather than
    requiring the caller to know each quantity's raw scale."""
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def solve_axis(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...], axis: str, other_bits: dict[str, int],
    lut_weight: float, bram_weight: float,
) -> tuple[dict[str, int], float, float]:
    """One MIP: pick a bit-width per stage on `axis` ('weight' or 'act'),
    minimizing normalized_sensitivity + lut_weight*normalized_lut +
    bram_weight*normalized_bram (all three CALIBRATED per finn_cost_model.py's
    own derating factors). No hard resource constraint at all -- see module
    docstring for why LUT lost its former hard budget once calibrated (it's
    exactly as chronically over-budget as BRAM already was)."""
    sens_key = "sensitivity_w" if axis == "weight" else "sensitivity_a"

    raw_sensitivity = {
        (s, b): sensitivity[s][sens_key][str(b)] for s in stage_names for b in CANDIDATE_BITS
    }
    raw_lut = {
        (s, b): stage_costs_for_axis(finn_costs, s, b, axis, other_bits, metric="total_lut")
        for s in stage_names for b in CANDIDATE_BITS
    }
    raw_bram = {
        (s, b): stage_costs_for_axis(finn_costs, s, b, axis, other_bits, metric="bram18k")
        for s in stage_names for b in CANDIDATE_BITS
    }
    sens_norm = _normalize(raw_sensitivity)
    lut_norm = _normalize(raw_lut)
    bram_norm = _normalize(raw_bram)

    prob = pulp.LpProblem(f"HAWQ_stage_bits_{axis}", pulp.LpMinimize)
    x = {
        (s, b): pulp.LpVariable(f"x_{axis}_{s}_{b}", cat=pulp.LpBinary)
        for s in stage_names for b in CANDIDATE_BITS
    }
    for s in stage_names:
        prob += pulp.lpSum(x[(s, b)] for b in CANDIDATE_BITS) == 1, f"one_bit_per_stage_{axis}_{s}"

    prob += pulp.lpSum(
        x[(s, b)] * (sens_norm[(s, b)] + lut_weight * lut_norm[(s, b)] + bram_weight * bram_norm[(s, b)])
        for s in stage_names for b in CANDIDATE_BITS
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"{axis} ILP did not solve to optimality (status={pulp.LpStatus[status]}).")
    result = {}
    for s in stage_names:
        chosen = [b for b in CANDIDATE_BITS if pulp.value(x[(s, b)]) > 0.5]
        assert len(chosen) == 1, f"stage {s} ({axis}): expected exactly one bit chosen, got {chosen}"
        result[s] = chosen[0]
    chosen_lut_total = sum(raw_lut[(s, result[s])] for s in stage_names)
    chosen_bram_total = sum(raw_bram[(s, result[s])] for s in stage_names)
    return result, chosen_lut_total, chosen_bram_total


XCZU7EV_LUT = 230_400  # real chip budget, see finn_cost_model.py's own docstring
XCZU7EV_BRAM_18K = 624  # real chip budget, see finn_cost_model.py's own docstring


def solve_stage_bits(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...], lut_weight: float, bram_weight: float,
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
    stage_weight_bits, w_lut, w_bram = solve_axis(
        sensitivity, finn_costs, stage_names, "weight", default_act, lut_weight, bram_weight,
    )
    stage_act_bits, a_lut, a_bram = solve_axis(
        sensitivity, finn_costs, stage_names, "act", stage_weight_bits, lut_weight, bram_weight,
    )
    # NOT w_lut+a_lut / w_bram+a_bram -- those are partial sums computed
    # under two DIFFERENT "other axis" assumptions (the weight pass assumed
    # act=8 everywhere via default_act; the act pass used the just-solved
    # real weight bits), so adding them double-counts under inconsistent bit
    # assumptions. The real combined totals at the actual final (w,a) pair
    # per stage, both CALIBRATED:
    total_lut = sum(
        _finn_cost(finn_costs, s, stage_weight_bits[s], stage_act_bits[s], "total_lut") for s in stage_names
    )
    total_bram = sum(
        _finn_cost(finn_costs, s, stage_weight_bits[s], stage_act_bits[s], "bram18k") for s in stage_names
    )
    return {
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "_diagnostics": {
            "total_lut_calibrated": total_lut,
            "xczu7ev_lut_budget": XCZU7EV_LUT,
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV_LUT,
            "total_bram18k_calibrated": total_bram,
            "xczu7ev_bram18k_budget": XCZU7EV_BRAM_18K,
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV_BRAM_18K,
            "note": "LUT and BRAM are BOTH a penalty in the objective (lut_weight/bram_weight), NOT a "
                    "hard constraint -- even the cheapest bit-width choice is well over budget on both, "
                    "once calibrated against this repo's real synthesis data point. See ilp_search.py's "
                    "own module docstring and finn_cost_model.py's LUT_DERATING_FACTOR/BRAM_DERATING_FACTOR "
                    "comment. These numbers are informational, already calibrated (not raw model output).",
        },
    }


def solve_joint_bits(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...], lut_weight: float, bram_weight: float,
) -> dict:
    """One combined MIP: pick a (weight_bit, act_bit) PAIR per stage jointly,
    instead of solve_stage_bits's two independent per-axis passes
    (solve_axis called once for weight holding act at a default-8 guess,
    then once for act holding weight at the just-solved result). That
    two-pass order is only an APPROXIMATION of the true joint cost --
    finn_cost_model.py's mvu_lut term genuinely depends on W*A jointly (it's
    not separable), so the weight pass's "hold act=8" assumption can look up
    a real cost number for a (w, 8) pair that never actually gets built once
    the act pass later picks a different act bit. This function removes that
    approximation entirely: it enumerates all 9 (w, a) candidate PAIRS per
    stage (CANDIDATE_BITS x CANDIDATE_BITS) and looks up finn_costs[s][f"W{w}
    _A{a}"] at the EXACT pair being considered, so the LUT/BRAM term is
    always the real joint cost, never an approximation.

    Sensitivity has no such joint measurement to fall back on though --
    sensitivity.py/block_sensitivity.py's quantization_deltas only ever
    perturbs ONE axis at a time (weight sensitivity is measured holding
    activations at full precision, and vice versa), so there is no directly
    -measured joint sensitivity(s, w, a) to look up. This uses the standard
    additive-independence approximation instead (sensitivity_w[s][w] +
    sensitivity_a[s][a]) -- the same approximation HAWQ-style per-layer ILP
    formulations commonly make when combining independently-measured
    per-axis Hessian-trace sensitivities into one joint score. So: this is
    now EXACT on the LUT/BRAM side and still APPROXIMATE on the sensitivity
    side -- strictly better than the two-pass method (which was approximate
    on BOTH sides), not a full joint solution.

    `stage_names` may be 5 stage-group names or dozens of individual
    bottleneck-block names, same as solve_stage_bits."""
    candidate_pairs = tuple((w, a) for w in CANDIDATE_BITS for a in CANDIDATE_BITS)

    raw_sensitivity = {
        (s, w, a): sensitivity[s]["sensitivity_w"][str(w)] + sensitivity[s]["sensitivity_a"][str(a)]
        for s in stage_names for w, a in candidate_pairs
    }
    raw_lut = {
        (s, w, a): _finn_cost(finn_costs, s, w, a, "total_lut")
        for s in stage_names for w, a in candidate_pairs
    }
    raw_bram = {
        (s, w, a): _finn_cost(finn_costs, s, w, a, "bram18k")
        for s in stage_names for w, a in candidate_pairs
    }
    sens_norm = _normalize(raw_sensitivity)
    lut_norm = _normalize(raw_lut)
    bram_norm = _normalize(raw_bram)

    prob = pulp.LpProblem("HAWQ_joint_bits", pulp.LpMinimize)
    x = {
        (s, w, a): pulp.LpVariable(f"x_{s}_{w}_{a}", cat=pulp.LpBinary)
        for s in stage_names for w, a in candidate_pairs
    }
    for s in stage_names:
        prob += pulp.lpSum(x[(s, w, a)] for w, a in candidate_pairs) == 1, f"one_pair_per_stage_{s}"

    prob += pulp.lpSum(
        x[(s, w, a)] * (sens_norm[(s, w, a)] + lut_weight * lut_norm[(s, w, a)] + bram_weight * bram_norm[(s, w, a)])
        for s in stage_names for w, a in candidate_pairs
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"joint ILP did not solve to optimality (status={pulp.LpStatus[status]}).")

    stage_weight_bits: dict[str, int] = {}
    stage_act_bits: dict[str, int] = {}
    for s in stage_names:
        chosen = [(w, a) for w, a in candidate_pairs if pulp.value(x[(s, w, a)]) > 0.5]
        assert len(chosen) == 1, f"stage {s}: expected exactly one (w,a) pair chosen, got {chosen}"
        stage_weight_bits[s], stage_act_bits[s] = chosen[0]

    total_lut = sum(raw_lut[(s, stage_weight_bits[s], stage_act_bits[s])] for s in stage_names)
    total_bram = sum(raw_bram[(s, stage_weight_bits[s], stage_act_bits[s])] for s in stage_names)
    return {
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "_diagnostics": {
            "total_lut_calibrated": total_lut,
            "xczu7ev_lut_budget": XCZU7EV_LUT,
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV_LUT,
            "total_bram18k_calibrated": total_bram,
            "xczu7ev_bram18k_budget": XCZU7EV_BRAM_18K,
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV_BRAM_18K,
            "note": "JOINT (w,a)-pair search (--joint): LUT/BRAM looked up at the exact chosen (w,a) "
                    "pair (no two-pass approximation), sensitivity is the additive sensitivity_w+"
                    "sensitivity_a approximation (see solve_joint_bits docstring). Both a penalty in "
                    "the objective, NOT a hard constraint -- same as the two-pass solve_stage_bits.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensitivity-file", type=Path, default=Path("compression/hawq/sensitivity_23_1.json"))
    parser.add_argument("--finn-cost-file", type=Path, default=Path("compression/hawq/finn_stage_costs.json"))
    parser.add_argument("--lut-weight", type=float, default=1.0,
                         help="Weight on normalized (calibrated) LUT cost in the objective, added to "
                              "normalized sensitivity (both in [0,1]) -- 0 disables LUT entirely. See "
                              "module docstring: this steers toward the lower end of LUT usage, it does "
                              "NOT guarantee fitting the real budget.")
    parser.add_argument("--bram-weight", type=float, default=1.0,
                         help="Same as --lut-weight, for calibrated BRAM_18K cost.")
    parser.add_argument("--joint", action="store_true",
                         help="Solve weight and activation bits TOGETHER as one (w,a)-pair MIP "
                              "(solve_joint_bits) instead of the default two-pass per-axis search "
                              "(solve_stage_bits). Looks up LUT/BRAM at the exact chosen (w,a) pair -- "
                              "no more two-pass 'hold the other axis at a default' approximation on "
                              "the resource-cost side. See solve_joint_bits's own docstring for what's "
                              "still approximate (the sensitivity side, additive by necessity).")
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

    if args.joint:
        result = solve_joint_bits(sensitivity, finn_costs, stage_names, args.lut_weight, args.bram_weight)
    else:
        result = solve_stage_bits(sensitivity, finn_costs, stage_names, args.lut_weight, args.bram_weight)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"stage_weight_bits: {result['stage_weight_bits']}")
    print(f"stage_act_bits:    {result['stage_act_bits']}")
    diag = result["_diagnostics"]
    print(f"LUT used (calibrated): {diag['total_lut_calibrated']:.0f} ({diag['lut_pct_of_budget']:.1f}% of {XCZU7EV_LUT} budget) -- informational.")
    print(f"BRAM_18K used (calibrated): {diag['total_bram18k_calibrated']:.0f} ({diag['bram_pct_of_budget']:.1f}% of {XCZU7EV_BRAM_18K} budget) -- informational.")


if __name__ == "__main__":
    main()
