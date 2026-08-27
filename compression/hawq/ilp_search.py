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

--joint also accepts --hard-lut/--hard-bram: add a real `<= XCZU7EV budget`
constraint instead of only the soft objective penalty described below (see
solve_joint_bits's own docstring for why this is --joint-only). Off by
default -- the objective-penalty formulation below remains the default
behavior for both --joint and the two-pass method.

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
repo's real Vivado synthesis data points (see finn_cost_model.py's own
_LUT_ANCHOR_FACTORS/_BRAM_ANCHOR_FACTORS module comment for the full
derivation, including a same-day correction of an initial mistake -- read
that comment, not just this summary). At avg_bits=8 (uniform W8A8, the
whole real 8-way design), real LUT usage was ~8.2x this model's own raw
estimate -- badly infeasible as a hard constraint. But a SECOND real data
point (partition_2 of a real per-block HAWQ build, using its own REAL
per-block bits AND real per-layer folding, LUT-weighted avg_bits=3.52 --
NOT uniform W2A2, an earlier version of this note got that wrong) showed
the derating factor falls sharply at lower bit-width -- only ~1.2x at
avg_bits=3.52, not ~8x -- so a low-bit-heavy HAWQ assignment is NOT nearly
as chronically over-budget as the old flat-8.2x calibration made it look
(this file's own calibrated_lut/calibrated_bram18k calls are now bit-
width-dependent, via avg_bits=(weight_bits+act_bits)/2 per unit, not a
single global multiplier -- clamped to [3.52, 8], the real measured range,
no extrapolation below it). Still kept as a PENALTY rather than a hard
constraint though, for a different reason now: this is genuinely only two
calibration points (avg_bits=3.52 and 8, nothing verified at 4 or below
3.52, nothing verified at a mixed-bits-in-one-partition granularity finer
than "one real per-block assignment on one real partition"), so treating
calibrated totals as a hard wall would bet the search's feasibility on an
interpolated curve shape that hasn't been checked, rather than as the
steering signal it actually is. Both resources are treated consistently:
penalties in the objective, using the CALIBRATED (derated, bit-width-aware)
totals so --lut-weight/--bram-weight steer toward genuinely cheaper designs
rather than an estimate that both gets the two resources' relative scale
wrong AND treats every bit-width the same. This also makes this file's own
formulation consistent with compression/hawq/folding_ilp.py's (which
already treats both as penalties, and is now also bit-width-aware the same
way). Neither weight "guarantees" fitting the real FPGA -- see
finn_cost_model.py's own calibration comment for why that claim can't be
made from two calibration points, and because folding (a separate axis,
see folding_ilp.py) is the real lever for actually reducing resource use,
not bit-width alone.

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

# 16 added 2026-08-25 -- every cost/sensitivity formula in this pipeline is
# purely parametric in weight_bits/act_bits (confirmed: finn_cost_model.py's
# mvu_lut/swu_bram18/wm_bram18 terms and sensitivity.py's own
# fake_quantize_symmetric all take bits as a free integer, no lookup table
# or bit-specific special-casing anywhere), so this generalizes cleanly.
# NOTE: finn_cost_model.py's own calibration (_LUT_ANCHOR_FACTORS/
# _BRAM_ANCHOR_FACTORS) is clamped to [3.52, 8] avg_bits -- there is no real
# synthesis data above avg_bits=8, so a 16-bit block's RAW cost (which DOES
# scale up correctly, e.g. mvu_lut's W*A term) still gets calibrated at the
# avg_bits=8 anchor factor, not extrapolated further. Same "don't
# extrapolate past the measured range" principle already applied at the low
# end -- not a bug, just an acknowledged approximation.
CANDIDATE_BITS = (2, 4, 8, 16)


def _finn_cost(finn_costs: dict, stage: str, weight_bits: int, act_bits: int, metric: str) -> float:
    entry = finn_costs[stage][f"W{weight_bits}_A{act_bits}"]
    if metric == "bram18k":
        return calibrated_bram18k(entry["swu_bram18"] + entry["wm_bram18"], weight_bits, act_bits)
    if metric == "total_lut":
        return calibrated_lut(entry["total_lut"], weight_bits, act_bits)
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
    lut_weight: float, bram_weight: float, candidate_bits: tuple[int, ...] | None = None,
    sensitivity_weight: float = 1.0,
) -> tuple[dict[str, int], float, float]:
    """One MIP: pick a bit-width per stage on `axis` ('weight' or 'act'),
    minimizing sensitivity_weight*normalized_sensitivity +
    lut_weight*normalized_lut + bram_weight*normalized_bram (all three
    CALIBRATED per finn_cost_model.py's own derating factors). No hard
    resource constraint at all -- see module docstring for why LUT lost its
    former hard budget once calibrated (it's exactly as chronically
    over-budget as BRAM already was).

    candidate_bits: the bit-widths actually offered to the solver for THIS
    axis -- defaults to the full module CANDIDATE_BITS, but a caller can
    pass a restricted subset (e.g. excluding 2 to enforce a minimum
    activation bit-width) to rule out a candidate entirely rather than just
    penalizing it. Note this only excludes it as an OPTION; it does not
    change the sensitivity/cost numbers for the bits still offered.

    sensitivity_weight: multiplier on the accuracy-impact term, default 1.0.
    Since only RATIOS between the three weights matter to the ILP's chosen
    minimizer (uniformly rescaling the whole objective never changes which
    candidate wins), this is mathematically redundant with just dividing
    lut_weight/bram_weight by the same factor -- it exists purely so
    "weight accuracy more/less" is one direct dial instead of two knobs
    that have to be moved in lockstep."""
    bits = candidate_bits if candidate_bits is not None else CANDIDATE_BITS
    sens_key = "sensitivity_w" if axis == "weight" else "sensitivity_a"

    raw_sensitivity = {
        (s, b): sensitivity[s][sens_key][str(b)] for s in stage_names for b in bits
    }
    raw_lut = {
        (s, b): stage_costs_for_axis(finn_costs, s, b, axis, other_bits, metric="total_lut")
        for s in stage_names for b in bits
    }
    raw_bram = {
        (s, b): stage_costs_for_axis(finn_costs, s, b, axis, other_bits, metric="bram18k")
        for s in stage_names for b in bits
    }
    sens_norm = _normalize(raw_sensitivity)
    lut_norm = _normalize(raw_lut)
    bram_norm = _normalize(raw_bram)

    prob = pulp.LpProblem(f"HAWQ_stage_bits_{axis}", pulp.LpMinimize)
    x = {
        (s, b): pulp.LpVariable(f"x_{axis}_{s}_{b}", cat=pulp.LpBinary)
        for s in stage_names for b in bits
    }
    for s in stage_names:
        prob += pulp.lpSum(x[(s, b)] for b in bits) == 1, f"one_bit_per_stage_{axis}_{s}"

    prob += pulp.lpSum(
        x[(s, b)] * (sensitivity_weight * sens_norm[(s, b)] + lut_weight * lut_norm[(s, b)] + bram_weight * bram_norm[(s, b)])
        for s in stage_names for b in bits
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"{axis} ILP did not solve to optimality (status={pulp.LpStatus[status]}).")
    result = {}
    for s in stage_names:
        chosen = [b for b in bits if pulp.value(x[(s, b)]) > 0.5]
        assert len(chosen) == 1, f"stage {s} ({axis}): expected exactly one bit chosen, got {chosen}"
        result[s] = chosen[0]
    chosen_lut_total = sum(raw_lut[(s, result[s])] for s in stage_names)
    chosen_bram_total = sum(raw_bram[(s, result[s])] for s in stage_names)
    return result, chosen_lut_total, chosen_bram_total


XCZU7EV_LUT = 230_400  # real chip budget, see finn_cost_model.py's own docstring
XCZU7EV_BRAM_18K = 624  # real chip budget, see finn_cost_model.py's own docstring


def solve_stage_bits(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...], lut_weight: float, bram_weight: float,
    min_act_bits: int | None = None, sensitivity_weight: float = 1.0,
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
    need to (see module docstring).

    min_act_bits: if set, the activation-axis solve_axis call excludes any
    CANDIDATE_BITS value below this floor entirely (not just penalizes it) --
    e.g. min_act_bits=4 rules out 2-bit activations everywhere. Weight axis
    is never restricted by this. Not derived from evidence that 2-bit acts
    are inherently untrainable (26_5_w24 trains fine at 2-bit acts
    everywhere in its context stage) -- this is a lever to test whether a
    higher act-bit floor changes the picture for a specific failing config,
    not a settled fix.

    sensitivity_weight: see solve_axis's own docstring -- passed through
    unchanged to both the weight and act passes."""
    default_act = {s: CANDIDATE_BITS[-1] for s in stage_names}
    stage_weight_bits, w_lut, w_bram = solve_axis(
        sensitivity, finn_costs, stage_names, "weight", default_act, lut_weight, bram_weight,
        sensitivity_weight=sensitivity_weight,
    )
    act_candidate_bits = (
        tuple(b for b in CANDIDATE_BITS if b >= min_act_bits) if min_act_bits is not None else None
    )
    stage_act_bits, a_lut, a_bram = solve_axis(
        sensitivity, finn_costs, stage_names, "act", stage_weight_bits, lut_weight, bram_weight,
        candidate_bits=act_candidate_bits, sensitivity_weight=sensitivity_weight,
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
        "status": "Optimal",
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "_diagnostics": {
            "total_lut_calibrated": total_lut,
            "xczu7ev_lut_budget": XCZU7EV_LUT,
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV_LUT,
            "total_bram18k_calibrated": total_bram,
            "xczu7ev_bram18k_budget": XCZU7EV_BRAM_18K,
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV_BRAM_18K,
            "min_act_bits": min_act_bits,
            "sensitivity_weight": sensitivity_weight,
            "note": "LUT and BRAM are BOTH a penalty in the objective (lut_weight/bram_weight), NOT a "
                    "hard constraint -- solve_stage_bits (the two-pass method) doesn't support a hard "
                    "constraint at all, see solve_joint_bits's own docstring for why; pass --joint "
                    "--hard-lut/--hard-bram for that. Calibrated bit-width-aware (avg_bits-interpolated "
                    "between real avg_bits=3.52/avg_bits=8 synthesis anchors, not a flat factor) -- see "
                    "ilp_search.py's own module docstring and finn_cost_model.py's "
                    "_LUT_ANCHOR_FACTORS/_BRAM_ANCHOR_FACTORS comment. These numbers are informational, "
                    "already calibrated (not raw model output).",
        },
    }


def solve_joint_bits(
    sensitivity: dict, finn_costs: dict, stage_names: tuple[str, ...], lut_weight: float, bram_weight: float,
    hard_lut: bool = False, hard_bram: bool = False, min_act_bits: int | None = None,
    sensitivity_weight: float = 1.0,
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

    hard_lut/hard_bram: opt-in hard `<= XCZU7EV budget` constraints (default
    both off, matching this file's own soft-penalty-only history -- see
    module docstring for why that was the original choice). Only supported
    HERE, not in the two-pass solve_stage_bits: a hard constraint needs a
    real, non-approximated per-candidate cost to constrain against, and
    solve_joint_bits is the only one of the two that has that (the two-pass
    method's per-axis raw_lut/raw_bram are evaluated holding the OTHER axis
    at a default/already-decided value, not the final combination -- a hard
    constraint built on that would silently under- or over-constrain
    depending on how far the final act/weight choice drifts from the
    assumption). Now that finn_cost_model.py's calibration is bit-width-
    aware (see its own module comment), a hard constraint is no longer
    guaranteed-infeasible the way the original flat-8.225x calibration made
    it -- this is genuinely worth trying, PuLP/CBC will just report
    "Infeasible" if it isn't achievable, not silently produce a wrong
    answer.

    `stage_names` may be 5 stage-group names or dozens of individual
    bottleneck-block names, same as solve_stage_bits.

    min_act_bits: if set, excludes any (w, a) pair with a < min_act_bits
    from candidate_pairs entirely -- same floor semantics as
    solve_stage_bits's own min_act_bits, just applied to the joint pair
    enumeration instead of a separate act-only axis.

    sensitivity_weight: see solve_axis's own docstring -- same semantics,
    applied to the additive sensitivity_w+sensitivity_a approximation used
    here."""
    act_bits_allowed = (
        tuple(b for b in CANDIDATE_BITS if b >= min_act_bits) if min_act_bits is not None else CANDIDATE_BITS
    )
    candidate_pairs = tuple((w, a) for w in CANDIDATE_BITS for a in act_bits_allowed)

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

    if hard_lut:
        prob += pulp.lpSum(
            x[(s, w, a)] * raw_lut[(s, w, a)] for s in stage_names for w, a in candidate_pairs
        ) <= XCZU7EV_LUT, "hard_lut_budget"
    if hard_bram:
        prob += pulp.lpSum(
            x[(s, w, a)] * raw_bram[(s, w, a)] for s in stage_names for w, a in candidate_pairs
        ) <= XCZU7EV_BRAM_18K, "hard_bram_budget"

    prob += pulp.lpSum(
        x[(s, w, a)] * (sensitivity_weight * sens_norm[(s, w, a)] + lut_weight * lut_norm[(s, w, a)] + bram_weight * bram_norm[(s, w, a)])
        for s in stage_names for w, a in candidate_pairs
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_name = pulp.LpStatus[status]
    if status_name not in ("Optimal", "Infeasible"):
        raise RuntimeError(f"joint ILP hit an unexpected solver status: {status_name!r}.")

    if status_name != "Optimal":
        return {
            "status": status_name,
            "stage_weight_bits": {},
            "stage_act_bits": {},
            "_diagnostics": {
                "note": f"Solver status {status_name!r} -- hard_lut={hard_lut}, hard_bram={hard_bram}. "
                        "No bit assignment can satisfy the hard constraint(s) requested; see "
                        "XCZU7EV_LUT/XCZU7EV_BRAM_18K for the budgets and finn_cost_model.py's own "
                        "calibration comment for the cost model this was checked against.",
            },
        }

    stage_weight_bits: dict[str, int] = {}
    stage_act_bits: dict[str, int] = {}
    for s in stage_names:
        chosen = [(w, a) for w, a in candidate_pairs if pulp.value(x[(s, w, a)]) > 0.5]
        assert len(chosen) == 1, f"stage {s}: expected exactly one (w,a) pair chosen, got {chosen}"
        stage_weight_bits[s], stage_act_bits[s] = chosen[0]

    total_lut = sum(raw_lut[(s, stage_weight_bits[s], stage_act_bits[s])] for s in stage_names)
    total_bram = sum(raw_bram[(s, stage_weight_bits[s], stage_act_bits[s])] for s in stage_names)
    return {
        "status": status_name,
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "_diagnostics": {
            "total_lut_calibrated": total_lut,
            "xczu7ev_lut_budget": XCZU7EV_LUT,
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV_LUT,
            "total_bram18k_calibrated": total_bram,
            "xczu7ev_bram18k_budget": XCZU7EV_BRAM_18K,
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV_BRAM_18K,
            "hard_lut": hard_lut,
            "hard_bram": hard_bram,
            "min_act_bits": min_act_bits,
            "sensitivity_weight": sensitivity_weight,
            "note": "JOINT (w,a)-pair search (--joint): LUT/BRAM looked up at the exact chosen (w,a) "
                    "pair (no two-pass approximation), sensitivity is the additive sensitivity_w+"
                    "sensitivity_a approximation (see solve_joint_bits docstring). "
                    + ("hard_lut/hard_bram constraints were ENFORCED (see those fields) -- this "
                       "assignment is guaranteed to fit, under this cost model's own calibration."
                       if (hard_lut or hard_bram) else
                       "LUT/BRAM are a soft penalty in the objective, NOT a hard constraint."),
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
    parser.add_argument("--sensitivity-weight", type=float, default=1.0,
                         help="Weight on normalized sensitivity (accuracy-impact proxy) in the "
                              "objective -- default 1.0. Mathematically redundant with dividing "
                              "--lut-weight/--bram-weight by the same factor (only the RATIO between "
                              "the three terms matters to the ILP's chosen minimizer), but this is the "
                              "direct single dial for 'weight accuracy more/less' instead of moving two "
                              "knobs in lockstep -- e.g. --sensitivity-weight 2.0 with the default "
                              "--lut-weight/--bram-weight 1.0 makes accuracy twice as important as "
                              "either resource term.")
    parser.add_argument("--joint", action="store_true",
                         help="Solve weight and activation bits TOGETHER as one (w,a)-pair MIP "
                              "(solve_joint_bits) instead of the default two-pass per-axis search "
                              "(solve_stage_bits). Looks up LUT/BRAM at the exact chosen (w,a) pair -- "
                              "no more two-pass 'hold the other axis at a default' approximation on "
                              "the resource-cost side. See solve_joint_bits's own docstring for what's "
                              "still approximate (the sensitivity side, additive by necessity).")
    parser.add_argument("--hard-lut", action="store_true",
                         help="Add a hard `<= XCZU7EV_LUT` constraint (calibrated LUT) instead of only a "
                              "soft penalty. Requires --joint (see solve_joint_bits's own docstring for "
                              "why the two-pass method can't support this). May report Infeasible -- "
                              "that's a real answer, not a bug, and gets written to --out-file same as "
                              "an Optimal result would.")
    parser.add_argument("--hard-bram", action="store_true", help="Same as --hard-lut, for calibrated BRAM_18K.")
    parser.add_argument("--candidate-bits", type=str, default=None,
                         help="Comma-separated override for the module-level CANDIDATE_BITS (e.g. "
                              "'2,4,8') -- needed when --sensitivity-file/--finn-cost-file come from a "
                              "config whose own block_sensitivity_*.py/finn_block_costs_*.py run used a "
                              "different candidate set than this file's own default (2,4,8,16). E.g. "
                              "S19's own block_sensitivity_s19.json/finn_block_costs_23_1.json were "
                              "regenerated with 16-bit included, but block_sensitivity_5_6.json/"
                              "finn_block_costs_5_6.json (5.6) were never rerun and only have (2,4,8) --"
                              "using the wrong candidate set raises a KeyError looking up a bit-width "
                              "the sensitivity/cost file doesn't have, not a silent wrong answer. Same "
                              "module-global-override pattern block_sensitivity.py itself already uses "
                              "for a different reason (injecting CANDIDATE_BITS into sensitivity.py's own "
                              "globals) -- see that file's own top-of-module comment.")
    parser.add_argument("--min-act-bits", type=int, default=None,
                         help="Exclude any activation bit-width below this floor from the solver's "
                              "candidate set entirely (not just penalize it) -- e.g. --min-act-bits 4 "
                              "rules out 2-bit activations everywhere, on the two-pass act axis or the "
                              "--joint pair enumeration. Default: no floor (all of CANDIDATE_BITS "
                              "eligible). NOTE: this is a diagnostic lever, not a validated fix -- "
                              "26_5_w24 trains to a real dice=0.6806 with 2-bit activations in 100% of "
                              "its context-stage blocks, so '2-bit acts are too aggressive' is not "
                              "well-supported by what's been observed so far.")
    parser.add_argument("--fix-bits", action="append", default=None, dest="fix_bits",
                         help="'<block_name>=<bits>', repeatable -- pins a block's weight AND "
                              "activation bits to the given value and EXCLUDES it from the ILP's own "
                              "decision variables entirely (not just penalizes other choices). Unlike "
                              "--min-act-bits (which restricts one axis, network-wide), this removes a "
                              "SPECIFIC block from the search completely -- its sensitivity/cost values "
                              "no longer contribute to _normalize()'s global min-max range either, so "
                              "fixing an extreme block (e.g. one with an outsized sensitivity value that "
                              "would otherwise dominate the normalization scale) can meaningfully change "
                              "which bits get chosen for the remaining free blocks, not just remove one "
                              "degree of freedom. E.g. --fix-bits initial=8 --fix-bits down1=8.")
    parser.add_argument("--out-file", type=Path, default=Path("compression/hawq/stage_bits_23_1.json"))
    args = parser.parse_args()

    if (args.hard_lut or args.hard_bram) and not args.joint:
        raise ValueError("--hard-lut/--hard-bram require --joint -- see solve_joint_bits's own docstring for why.")

    if args.candidate_bits is not None:
        global CANDIDATE_BITS
        CANDIDATE_BITS = tuple(sorted(int(b) for b in args.candidate_bits.split(",")))
        print(f"Overriding CANDIDATE_BITS to {CANDIDATE_BITS} (from --candidate-bits).")

    if args.min_act_bits is not None and args.min_act_bits not in CANDIDATE_BITS:
        raise ValueError(f"--min-act-bits {args.min_act_bits} is not in the effective CANDIDATE_BITS {CANDIDATE_BITS}.")

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

    fixed_bits: dict[str, int] = {}
    if args.fix_bits:
        for spec in args.fix_bits:
            if "=" not in spec:
                raise ValueError(f"--fix-bits {spec!r} must be '<block_name>=<bits>'.")
            block_name, bits_str = spec.split("=", 1)
            if block_name not in stage_names:
                raise ValueError(f"--fix-bits {spec!r}: {block_name!r} not found in --sensitivity-file's own names.")
            fixed_bits[block_name] = int(bits_str)
        print(f"Fixing {len(fixed_bits)} block(s) outside the ILP search: {fixed_bits}")

    free_stage_names = tuple(s for s in stage_names if s not in fixed_bits)

    if args.joint:
        result = solve_joint_bits(
            sensitivity, finn_costs, free_stage_names, args.lut_weight, args.bram_weight,
            hard_lut=args.hard_lut, hard_bram=args.hard_bram, min_act_bits=args.min_act_bits,
            sensitivity_weight=args.sensitivity_weight,
        )
    else:
        result = solve_stage_bits(
            sensitivity, finn_costs, free_stage_names, args.lut_weight, args.bram_weight,
            min_act_bits=args.min_act_bits, sensitivity_weight=args.sensitivity_weight,
        )

    if fixed_bits and result["status"] == "Optimal":
        for block_name, bits in fixed_bits.items():
            result["stage_weight_bits"][block_name] = bits
            result["stage_act_bits"][block_name] = bits
        fixed_lut = sum(_finn_cost(finn_costs, b, bits, bits, "total_lut") for b, bits in fixed_bits.items())
        fixed_bram = sum(_finn_cost(finn_costs, b, bits, bits, "bram18k") for b, bits in fixed_bits.items())
        result["_diagnostics"]["total_lut_calibrated"] += fixed_lut
        result["_diagnostics"]["total_bram18k_calibrated"] += fixed_bram
        result["_diagnostics"]["lut_pct_of_budget"] = 100 * result["_diagnostics"]["total_lut_calibrated"] / XCZU7EV_LUT
        result["_diagnostics"]["bram_pct_of_budget"] = 100 * result["_diagnostics"]["total_bram18k_calibrated"] / XCZU7EV_BRAM_18K
        result["_diagnostics"]["fixed_bits"] = fixed_bits
        result["_diagnostics"]["note"] += (
            f" ADDITIONALLY: {fixed_bits} were pinned via --fix-bits and excluded from the ILP's own "
            f"decision variables and from _normalize()'s min-max range entirely -- the remaining "
            f"{len(free_stage_names)} blocks were solved as if this were the full candidate set. LUT/BRAM "
            f"totals above include the fixed blocks' own real (non-approximated) cost."
        )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"ILP status: {result['status']}")
    if result["status"] != "Optimal":
        print(f"No bit assignment satisfies the requested hard constraint(s) -- see {args.out_file}'s "
              f"own _diagnostics.note.")
        return
    print(f"stage_weight_bits: {result['stage_weight_bits']}")
    print(f"stage_act_bits:    {result['stage_act_bits']}")
    diag = result["_diagnostics"]
    budget_word = "guaranteed (hard constraint)" if (args.hard_lut or args.hard_bram) else "informational"
    print(f"LUT used (calibrated): {diag['total_lut_calibrated']:.0f} ({diag['lut_pct_of_budget']:.1f}% of {XCZU7EV_LUT} budget) -- {budget_word}.")
    print(f"BRAM_18K used (calibrated): {diag['total_bram18k_calibrated']:.0f} ({diag['bram_pct_of_budget']:.1f}% of {XCZU7EV_BRAM_18K} budget) -- {budget_word}.")


if __name__ == "__main__":
    main()
