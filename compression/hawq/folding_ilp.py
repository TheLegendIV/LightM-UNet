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
script's own earlier INFEASIBLE-at-W8A8 finding).

IMPORTANT CALIBRATION NOTE (updated 2026-08-25): this repo used to claim
"real Vivado synthesis typically comes in well under this closed-form
analytical LUT/BRAM estimate" -- that claim was NEVER actually verified
against real synthesis for this cost model, and turned out to be WRONG for
LUT once we got a real data point. Calibrating against S19's real
8-way-partitioned OOC Vivado synthesis (hardware/results.csv's
`s19_double_mid_8way_partitioned_ooc_synth_TOTAL` row, uniform W8A8) plus
the real resolved per-layer PE/SIMD folding config from that build
(hardware/outputs/s19_8way_partitioned_ooc_20260820_101224/final_hw_config.json
-- effectively FOLDING_SERIAL, PE=SIMD=1 on nearly every MVAU node) gave a
first calibration point:
    LUT:      830,689 real  vs. 100,996 this-model-at-FOLDING_SERIAL -> 8.225x OVER (whole design)
    BRAM_18K:     906 real  vs.   1,495 this-model-at-FOLDING_SERIAL -> 0.606x (under, whole design)
A SECOND real data point (hardware/results.csv's
`s19_hawq_block_partition_2_ooc_synth` row) is a real per-block HAWQ bit
assignment (the actual historical compression/hawq/block_bits_s19.json used
for that build, NOT necessarily whatever the file on disk holds now -- this
repo's own ILP has been rerun/recalibrated multiple times since, so the
bits/folding a real hardware build was synthesized with have to be pulled
from that build's own saved artifacts, not assumed from the current repo
state) on the largest 8-way partition (down2 + stage2.0-2.5.reduce.0, 23
real layers), evaluated at its own real per-layer folding
(hardware/outputs/s19_hawq_block_partition_2_ooc_synth_20260824_220316/
hawq_folding_config_partition2.json -- PE=1, SIMD in {4,6,8,12}, NOT
FOLDING_SERIAL). An EARLIER version of this note got both of those wrong
(assumed the partition's real bits were uniform W2A2, and evaluated this
model at the CURRENT/regenerated folding_block_s19.json instead of the
real historical folding) -- recomputing directly from this model's own
layer_cost_pe_simd(), fed the real per-block bits and real per-layer
(PE, SIMD) for all 23 layers, gives a LUT-weighted mean avg_bits=3.52
(NOT 2) for that partition, and a genuine second calibration anchor there.
A controlled comparison against that same partition's own uniform-W8A8
baseline row showed the derating factor is NOT constant with bit-width: it
falls sharply at lower bit-width (LUT ~8.2x at avg_bits=8 down to ~1.2x at
avg_bits=3.52; BRAM ~0.61x down to ~0.13x -- see finn_cost_model.py's own
`_LUT_ANCHOR_FACTORS`/`_BRAM_ANCHOR_FACTORS` comment for the exact numbers
and reasoning). `calibrated_lut`/`calibrated_bram18k` in finn_cost_model.py
are now BIT-WIDTH-DEPENDENT (linear interpolation between the avg_bits=3.52
and avg_bits=8 anchors at avg_bits=(weight_bits+act_bits)/2, CLAMPED to
[3.52, 8] -- no extrapolation below the real lower anchor), applied HERE at
each layer's own resolved bits (both when building the objective's
lut_norm/bram_norm below, and again in `_diagnostics`) -- NOT a flat
multiplier applied to an aggregate sum. This is still only two calibration
points (nothing verified at 4-bit, nothing verified below avg_bits=3.52, or
at a finer per-partition granularity), so treat calibrated totals as a
steering signal, not a guarantee. cycles/LUT/BRAM
are on wildly different scales (cycles range from ~1e2 to ~1e8 per
candidate depending on folding; LUT ~1e2-1e5; BRAM ~1e0-1e3), so each is
independently min-max normalized to [0,1] over the FULL (layer, pe, simd)
candidate space before combining -- normalized on the CALIBRATED per-layer
LUT/BRAM (not raw), since the bit-width-dependent factor is no longer a
normalization-invariant global constant: two layers can now differ in
calibrated magnitude by their bit-width alone, and the objective should see
that -- same convention ilp_search.py's own `_normalize` uses for
sensitivity/BRAM, just extended to a 3rd term here:
    minimize sum_{l,pe,simd} x[l,pe,simd] * (cycles_norm[l,pe,simd]
                                              + lut_weight * lut_norm[l,pe,simd]
                                              + bram_weight * bram_norm[l,pe,simd])
Constraint (the only hard one, by default): sum_{pe,simd} x[layer,pe,simd]
== 1 per layer -- exactly one folding choice per layer. No LUT/BRAM `<=`
constraint by default -- pass --hard-lut/--hard-bram to add one anyway (a
real `<= FRACTION * XCZU7EV['LUT'/'BRAM_18K']` constraint on the same
per-layer CALIBRATED totals used everywhere else in this file). Bare
--hard-lut/--hard-bram (no value) means FRACTION=1.0 (the full nominal
budget); pass e.g. --hard-lut 0.7 --hard-bram 0.8 to cap under 100%,
reserving headroom for DSP/FF -- resources this cost model does NOT
estimate at all (see "Not covered here" below) -- since there's currently
no way to constrain those directly. Off by default; may report Infeasible
(a real answer, not an error).

Chosen LUT/BRAM/cycles totals (both raw and calibrated) are still computed
and reported after solving (both on stdout and in the output JSON's
"_diagnostics") -- informational unless --hard-lut/--hard-bram was passed,
in which case an Optimal result's totals are a guarantee, not just a
steering signal, per the module's own reasoning above.

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
from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from finn_cost_model import (  # noqa: E402
    RAM_STYLE_BLOCK, RAM_STYLE_ULTRA, LayerGeometry, calibrated_bram18k, calibrated_lut, divisors,
    layer_cost_pe_simd, max_pe, max_simd,
)
from finn_stage_costs import INPUT_HW, dump_layer_geometry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}
# No established URAM_18K budget constant for the XCZU7EV yet (unlike LUT/
# BRAM_18K above) -- total_uram18 is still reported in solve_folding's own
# _diagnostics, just without a "% of budget" figure until a real number is
# sourced (see finn_cost_model.py's own RAM_STYLE_ULTRA docstring for the
# wm_uram18 formula this is summing).
RAM_STYLES = (RAM_STYLE_BLOCK, RAM_STYLE_ULTRA)


def load_config(config_module: str) -> None:
    """Same pattern as sensitivity.py/finn_stage_costs.py's own loader --
    injects the named config_*.py's constants into this module's globals,
    overriding the static config_23_1 default imported above."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def candidate_folds(layer: LayerGeometry) -> list[tuple[int, int, str]]:
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
    lut_weight: float, bram_weight: float, hard_lut: float | None = None, hard_bram: float | None = None,
) -> dict:
    """hard_lut/hard_bram: None (default) = no hard constraint, matching
    this file's original soft-penalty-only behavior. A float `f` in (0, 1]
    adds a hard `<= f * XCZU7EV[...]` constraint -- e.g. hard_lut=0.7 caps
    calibrated total LUT at 70% of the real chip budget, not 100%. Lets a
    caller reserve headroom for resources this cost model doesn't estimate
    at all (DSP, FF -- see module docstring's own "Not covered here" list)
    by under-capping the ones it DOES estimate, rather than only being able
    to ask for the full nominal LUT/BRAM budget."""
    x = {}
    layer_costs: dict[str, dict[tuple[int, int, str], dict]] = {}
    raw_cycles: dict[tuple[str, int, int, str], float] = {}
    raw_lut: dict[tuple[str, int, int, str], float] = {}
    raw_bram: dict[tuple[str, int, int, str], float] = {}

    for layer in geometries:
        w_bits, a_bits = layer_bits(layer, stage_bits, weight_bits, act_bits)
        folds = candidate_folds(layer)
        costs = {
            (pe, simd, ram_style): layer_cost_pe_simd(layer, w_bits, a_bits, pe, simd, ram_style)
            for pe, simd, ram_style in folds
        }
        layer_costs[layer.name] = costs
        for pe, simd, ram_style in folds:
            fold_key = (pe, simd, ram_style)
            key = (layer.name, pe, simd, ram_style)
            # URAM (wm_uram18) is a separate FPGA resource from BRAM_18K --
            # deliberately NOT folded into raw_bram (would misreport a
            # ram_style="ultra" choice as costing BRAM it doesn't actually
            # use). See module docstring: no established URAM budget
            # constant yet, so it's summed in _diagnostics without a "% of
            # budget" figure, same honesty-over-fabrication choice as
            # everywhere else in this file.
            raw_cycles[key] = costs[fold_key]["cycles"]
            # Calibrated at THIS layer's own (w_bits, a_bits) -- calibrated_lut/
            # calibrated_bram18k are bit-width-dependent (interpolated between
            # real avg_bits=3.52/avg_bits=8 synthesis anchors, see
            # finn_cost_model.py), so unlike the old flat-factor version,
            # applying this per layer BEFORE normalizing actually changes the
            # relative normalized magnitudes across layers with different
            # bit-widths -- no longer a normalization-invariant no-op.
            raw_lut[key] = calibrated_lut(costs[fold_key]["total_lut"], w_bits, a_bits)
            raw_bram[key] = calibrated_bram18k(costs[fold_key]["swu_bram18"] + costs[fold_key]["wm_bram18"], w_bits, a_bits)
            x[key] = pulp.LpVariable(f"x_{layer.name}_{pe}_{simd}_{ram_style}", cat=pulp.LpBinary)

    cycles_norm = _normalize(raw_cycles)
    lut_norm = _normalize(raw_lut)
    bram_norm = _normalize(raw_bram)

    prob = pulp.LpProblem("HAWQ_folding", pulp.LpMinimize)
    for layer in geometries:
        folds = candidate_folds(layer)
        prob += pulp.lpSum(x[(layer.name, pe, simd, ram_style)] for pe, simd, ram_style in folds) == 1, f"one_fold_{layer.name}"

    # LUT/BRAM are a PENALTY here by default, not a hard `<=budget`
    # constraint -- see module docstring for why (fully-unfolded LUT/BRAM is
    # often over budget regardless of folding choice, and a hard constraint
    # on a still-only-two-point-calibrated, bit-width-dependent estimate
    # risks false infeasibility either direction -- calibration shows this
    # model UNDER-shoots LUT and OVER-shoots BRAM at every bit-width checked
    # so far, avg_bits=3.52 through avg_bits=8, just by very different
    # margins depending on bit-width). hard_lut/hard_bram (opt-in, see
    # ilp_search.py's own solve_joint_bits docstring for the same rationale
    # applied there) add a real `<=` constraint on top instead of just
    # steering the objective -- raw_lut/raw_bram here are already the
    # per-layer CALIBRATED values (see above), so this constrains the same
    # bit-width-aware totals reported in "_diagnostics", not the raw model
    # output. May report Infeasible -- handled below like any other status,
    # not treated as an error.
    if hard_lut is not None:
        prob += pulp.lpSum(x[key] * raw_lut[key] for key in x) <= hard_lut * XCZU7EV["LUT"], "hard_lut_budget"
    if hard_bram is not None:
        prob += pulp.lpSum(x[key] * raw_bram[key] for key in x) <= hard_bram * XCZU7EV["BRAM_18K"], "hard_bram_budget"

    prob += pulp.lpSum(
        x[key] * (cycles_norm[key] + lut_weight * lut_norm[key] + bram_weight * bram_norm[key])
        for key in x
    )

    # timeLimit/gapRel: a real, permanent safety net -- a hard_lut/hard_bram
    # constraint sitting close to the true optimum (rather than comfortably
    # slack, as the unconstrained-at-100%-cap cases were) makes CBC's
    # branch-and-bound tree MUCH harder to fully close out: confirmed
    # empirically, a --hard-lut 0.70 --hard-bram 0.80 S19 block-granularity
    # solve ran over an hour of real CPU time with no plain PULP_CBC_CMD()
    # call ever returning. gapRel=0.01 (accept anything CBC can PROVE is
    # within 1% of the true optimum) plus a 300s hard wall-clock cap fixes
    # this -- CBC almost always finds a very-near-optimal incumbent within
    # seconds, the remaining time is normally spent proving no better
    # solution exists, which this exploratory HAWQ search doesn't need.
    # status still comes back "Optimal" if the gap or time limit is what
    # stopped it (not a genuine proof) -- PuLP/CBC doesn't distinguish that
    # in LpStatus, so a result here is "very likely optimal, not certified,"
    # not literally provably exact whenever a hard_lut/hard_bram constraint
    # is anywhere close to binding.
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300, gapRel=0.01))
    status_name = pulp.LpStatus[status]
    result_per_layer = {}
    if status_name == "Optimal":
        for layer in geometries:
            for pe, simd, ram_style in candidate_folds(layer):
                if pulp.value(x[(layer.name, pe, simd, ram_style)]) > 0.5:
                    w_bits, a_bits = layer_bits(layer, stage_bits, weight_bits, act_bits)
                    result_per_layer[layer.name] = {
                        "stage": layer.stage, "pe": pe, "simd": simd, "ram_style": ram_style,
                        "weight_bits": w_bits, "act_bits": a_bits,
                        **layer_costs[layer.name][(pe, simd, ram_style)],
                    }
                    break

    total_lut_raw = sum(v["total_lut"] for v in result_per_layer.values())
    total_bram_raw = sum(v["swu_bram18"] + v["wm_bram18"] for v in result_per_layer.values())
    total_uram = sum(v.get("wm_uram18", 0) for v in result_per_layer.values())
    total_cycles = sum(v["cycles"] for v in result_per_layer.values())
    # Calibrated PER LAYER at its own chosen (weight_bits, act_bits) -- see
    # finn_cost_model.py's own _LUT_ANCHOR_FACTORS/_BRAM_ANCHOR_FACTORS
    # comment. NOT total_lut_raw run through one flat multiplier: the
    # derating factor is bit-width-dependent now, so a heterogeneous-bit-
    # width layer set has to be calibrated layer-by-layer and then summed,
    # not summed-then-calibrated (those give different answers whenever
    # layers disagree on bit-width, which per-block HAWQ configs always do).
    # Doesn't change WHICH folding the ILP picks for a FIXED bit assignment
    # (min-max normalization inside the objective above is invariant to a
    # per-layer-constant rescaling that doesn't depend on pe/simd/ram_style),
    # only what's reported here as "% of budget".
    total_lut = sum(calibrated_lut(v["total_lut"], v["weight_bits"], v["act_bits"]) for v in result_per_layer.values())
    total_bram = sum(
        calibrated_bram18k(v["swu_bram18"] + v["wm_bram18"], v["weight_bits"], v["act_bits"])
        for v in result_per_layer.values()
    )
    return {
        "status": status_name,
        "per_layer": result_per_layer,
        "_diagnostics": {
            "total_lut_calibrated": total_lut, "total_lut_raw": total_lut_raw,
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k_calibrated": total_bram, "total_bram18k_raw": total_bram_raw,
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "total_uram18": total_uram,
            "total_cycles": total_cycles,
            "hard_lut": hard_lut,
            "hard_bram": hard_bram,
            "note": ("hard_lut/hard_bram constraints were ENFORCED (see those fields, each either null "
                     "or the fraction of XCZU7EV's nominal budget actually capped at) -- if status is "
                     "Optimal, this folding is GUARANTEED to fit those (possibly fractional) budgets "
                     "under this cost model's own calibration; if Infeasible, no folding choice can "
                     "satisfy the requested constraint(s) for this bit assignment. "
                     if (hard_lut is not None or hard_bram is not None) else
                     "LUT/BRAM are informational only here (penalty in the objective via "
                     "--lut-weight/--bram-weight, NOT a hard constraint), ")
                    + "already calibrated "
                    "PER LAYER at its own bit-width against this repo's real synthesis data points -- "
                    "see finn_cost_model.py's own _LUT_ANCHOR_FACTORS/_BRAM_ANCHOR_FACTORS comment and "
                    "this file's own module docstring. Over 100% does not mean the search failed; "
                    "_raw fields are the uncalibrated model output, for reference.",
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
    parser.add_argument("--hard-lut", type=float, nargs="?", const=1.0, default=None, metavar="FRACTION",
                         help="Add a hard `<= FRACTION * XCZU7EV['LUT']` constraint (calibrated, per-layer "
                              "bit-width-aware) on top of the objective, instead of only a soft penalty. "
                              "Bare --hard-lut (no value) means FRACTION=1.0 (the full nominal budget) -- "
                              "pass e.g. --hard-lut 0.7 to cap at 70% instead, reserving headroom for "
                              "resources this cost model doesn't estimate at all (DSP, FF -- see module "
                              "docstring). May report Infeasible -- that's a real answer, not a bug, and "
                              "gets written to --out-file same as an Optimal result would.")
    parser.add_argument("--hard-bram", type=float, nargs="?", const=1.0, default=None, metavar="FRACTION",
                         help="Same as --hard-lut, for calibrated BRAM_18K.")
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

    # OPTIONAL config globals -- see sensitivity.py's own build_fp32_model
    # for the full rationale (not repeated here): every pre-S8.2 config_*.py
    # never defines these (PReLU + plain projected bottlenecks), so .get()
    # falls back to ENet.py's own defaults, unchanged for them.
    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=globals().get("REG_BOOKEND_DSC", False),
    )
    if args.granularity == "block":
        geometries, _block_names = dump_block_layer_geometry(model, INPUT_HW)
    else:
        geometries = dump_layer_geometry(model, INPUT_HW)
    n_candidates = sum(len(candidate_folds(g)) for g in geometries)
    print(f"Traced {len(geometries)} layers ({args.granularity} granularity), {n_candidates} total "
          f"(layer, PE, SIMD) candidates. "
          f"Bits: {'per-' + args.granularity + ' from ' + str(args.stage_bits_file) if stage_bits else f'uniform W{args.weight_bits}A{args.act_bits}'}. Solving...")

    result = solve_folding(
        geometries, stage_bits, args.weight_bits, args.act_bits, args.lut_weight, args.bram_weight,
        hard_lut=args.hard_lut, hard_bram=args.hard_bram,
    )
    print(f"ILP status: {result['status']}")
    if result["status"] == "Optimal":
        diag = result["_diagnostics"]
        print(f"Total LUT: {diag['total_lut_calibrated']:.0f} calibrated, {diag['total_lut_raw']:.0f} raw "
              f"({diag['lut_pct_of_budget']:.1f}% of {XCZU7EV['LUT']} budget)")
        print(f"Total BRAM_18K: {diag['total_bram18k_calibrated']:.0f} calibrated, {diag['total_bram18k_raw']:.0f} raw "
              f"({diag['bram_pct_of_budget']:.1f}% of {XCZU7EV['BRAM_18K']} budget)")
        print(f"Total cycles (sum, ~= per-image latency): {diag['total_cycles']:.0f}")
        if args.hard_lut is not None or args.hard_bram is not None:
            print(f"(hard_lut={args.hard_lut}, hard_bram={args.hard_bram} (fraction of nominal budget, "
                  f"None = no hard constraint) -- this folding is GUARANTEED to fit the requested "
                  f"budget(s) under this cost model's own calibration.)")
        else:
            print("(LUT/BRAM are calibrated soft penalties in the objective -- see module docstring -- "
                  "no hard budget was enforced.)")
    else:
        print(f"No folding choice satisfies the requested hard constraint(s) for this bit assignment -- "
              f"see {args.out_file}'s own _diagnostics.note.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
