"""Combined bits+folding MILP -- jointly picks per-BLOCK bit-width
(weight_bits, act_bits) AND per-LAYER folding (pe, simd, ram_style) in ONE
solve, instead of ilp_search.py's bit-search followed by folding_ilp.py's
folding-search taking those bits as fixed.

WHY: LUT/BRAM cost is a joint, non-separable function of (bits, folding) --
never a function of bits alone (finn_cost_model.py's mvu_lut/wm_bram18
formulas multiply fold and bit-width terms together). Confirmed on this
repo's own S12 min4 sweep: block_bits_12_separable_dense_relu_min4_joint_
sw0.3.json reports 450.5% over LUT budget against ilp_search.py's own
FOLDING_UNFOLDED-assumption cost model -- yet folds to a real Optimal,
86.03% LUT, under `folding_ilp.py --force-serial --hard-lut 1.0`. sw0.5
reports similarly "over" but is a genuine Infeasible once real folding is
checked. The bits-stage's own feasibility signal has no reliable
correlation with real feasibility. This script removes the two-stage seam
entirely: LUT and BRAM are REAL hard `<=` constraints here (not a soft
penalty, unlike both ilp_search.py's and folding_ilp.py's own defaults),
and the objective is a single `alpha`/`(1-alpha)` dial between the
sensitivity (accuracy) proxy and latency (cycles) -- no separate lut_weight/
bram_weight knobs, since LUT/BRAM are constraints here, not objective terms.

VARIABLES:
    y[block, w, a]   one-hot per block (sum_{w,a} y[block,w,a] == 1) --
                     matches ilp_search.py's solve_joint_bits exactly.
                     Sensitivity attaches here.
    z[layer, pe, simd, ram_style, w, a]   one-hot per layer, over
                     candidate_folds(layer) x candidate_pairs -- one binary
                     per (layer, fold, bit-pair) combination. Cycles/LUT/
                     BRAM attach here, at the EXACT (fold, bits) each layer
                     would actually run at -- this is the minimal exact
                     linearization of the bilinear cost formulas (a binary
                     per full combination avoids needing McCormick auxiliary
                     variables to multiply a separate bits variable by a
                     separate fold variable).

LINKING CONSTRAINT (same equality-coupling idiom solve_folding_nodewise
already uses for its own FMPadding SIMD == SWU SIMD == VVAU PE chaining,
applied one level up -- block->layer instead of layer->sub-node):
    for layer, (w,a):  sum_{pe,simd,ram_style} z[layer,...,w,a] == y[layer.stage,w,a]
A block's chosen (w,a) forces every layer in it to y=1's (w,a) exclusively --
not "steered toward", mathematically infeasible to violate: at any OTHER
(w,a) pair the right side is 0, and since every z is binary and must sum to
exactly 0, each individual z at that pair is forced to 0 too. An
inconsistent assignment (a layer using different bits than its block chose)
is never in the feasible region, so nothing needs checking after solving.

OBJECTIVE -- rescaled by term COUNT, not just min-max [0,1] normalized (a
genuinely new issue this combined ILP introduces neither existing script
has: ilp_search.py's sens_norm/lut_norm/bram_norm and folding_ilp.py's
cycles_norm/lut_norm/bram_norm are always combined per-variable over the
SAME index set; here y (one active term per BLOCK, e.g. 29) and z (one
active term per LAYER, e.g. 104) are separate index sets combined for the
first time). A plain sum would make the latency term mechanically larger
just from having more nonzero terms, nothing to do with which matters more
-- biasing alpha=0.5 away from equal weight before the dial does anything.
Fixed by dividing each sum by its own (known, constant) term count:
    sensitivity_term = (1/n_blocks) * sum_{block,w,a} y[block,w,a] * sens_norm[block,w,a]
    latency_term     = (1/n_layers) * sum_{layer,fold,w,a} z[layer,fold,w,a] * cycles_norm[layer,fold,w,a]
    minimize alpha * sensitivity_term + (1-alpha) * latency_term
Dividing by a constant (not a variable) keeps this fully linear.

HARD CONSTRAINTS (the headline change from both existing scripts -- real
budget here, always on, not opt-in soft penalty):
    sum z*raw_lut  <= hard_lut_fraction  * XCZU7EV["LUT"]
    sum z*raw_bram <= hard_bram_fraction * XCZU7EV["BRAM_18K"]

Sensitivity has no true joint (w,a) measurement to fall back on (same
limitation ilp_search.py's own solve_joint_bits already documents) -- uses
the same additive approximation, sensitivity_w[block][w] + sensitivity_a[block][a].

Cycles do NOT depend on bit-width at all (conv_cost_pe_simd's cycles formula
has no W/A term) -- confirmed empirically: two DIFFERENT bit assignments
folded at --force-serial report identical total_cycles=136,138,752 for S12.
So bits influence latency only INDIRECTLY, by determining which (LUT/BRAM-
budget-respecting) folding choices are even affordable -- alpha=0 doesn't
select bits to directly minimize a bits-dependent cycle term (none exists),
it selects bits that most PERMIT low-cycle folding under the hard cap. This
is exactly the joint effect the two-stage pipeline can't see.

--force-serial (same mechanism/name as folding_ilp.py's own flag, reused
via that module's FORCE_SERIAL global + candidate_folds) restricts every
layer to the single (1,1,ram_style) FOLDING_SERIAL candidate before
solving -- used here mainly for VALIDATING this script against
folding_ilp.py's own already-computed answers (see module tests below), not
part of the normal alpha-sweep workflow.

Scope (see compression/hawq/joint_bits_folding_ilp.py's own plan doc,
C:\\Users\\win32\\.claude\\plans\\nested-singing-flurry.md, for the full
discussion): quantization stays per-BLOCK (matches CombinedQuantENet.py's
own block constructors, which take one (weight_bits, act_bits) pair per
block); folding stays layer-wise, not --node-level's 3-node depthwise split
(a confirmed no-op for S12 -- 0 depthwise layers).

Usage:
    python compression/hawq/joint_bits_folding_ilp.py \\
        --config config_12_separable_dense_relu \\
        --sensitivity-file compression/hawq/artifacts/block_sensitivity_12_separable_dense_relu.json \\
        --candidate-bits 4,8 --alpha 0.3 \\
        --out-file compression/hawq/artifacts/block_bits_folding_12_separable_dense_relu_joint_alpha0.3.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pulp
import torch  # noqa: F401 -- imported for side effect parity with folding_ilp.py's own model-tracing setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, IN_CHANNELS,
    OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, USE_ASYMMETRIC,
)
import folding_ilp as _folding  # noqa: E402 -- reused for candidate_folds/candidate_swu_simd/FORCE_SERIAL
from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from finn_cost_model import LayerGeometry, calibrated_bram18k, calibrated_lut, layer_cost_pe_simd  # noqa: E402
from finn_stage_costs import INPUT_HW, dump_layer_geometry  # noqa: E402
from ilp_search import _normalize  # noqa: E402 -- one source of truth for the [0,1] normalization helper

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}  # same values as ilp_search.py/folding_ilp.py's own copies
CANDIDATE_BITS = (2, 4, 6, 8, 16)


def load_config(config_module: str) -> None:
    """Same pattern as ilp_search.py/folding_ilp.py's own loader."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def solve_joint(
    sensitivity: dict, geometries: list[LayerGeometry], block_names: tuple[str, ...],
    alpha: float, hard_lut_fraction: float, hard_bram_fraction: float,
    time_limit: int, gap_rel: float, max_cycles: float | None = None,
    pinned_bits: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """The combined MILP -- see module docstring for the full formulation.

    max_cycles: optional hard `<=` constraint on total cycles (the SAME
    quantity `latency_term` in the objective already steers toward, just as
    a real cap now, not only a soft preference) -- e.g. a "no more than 2s
    @ 100MHz" requirement is max_cycles=2.0/1000*100e6=200_000_000. Makes
    EVERY alpha in [0,1] automatically respect the budget by construction
    (the solver finds the most-accurate assignment WITHIN the acceptable
    latency envelope at that alpha), rather than needing to sweep alpha,
    solve, discard whichever points exceed the budget, and re-sample --
    that approach also silently assumes a single clean alpha-vs-cycles
    crossing point, which isn't guaranteed in general. None (default) = no
    latency cap, matching this script's original behavior.

    pinned_bits: TEST-ONLY lever -- {block_name: (w, a)} -- when given, y is
    pinned to these exact values via equality constraints instead of being
    freely chosen (still a real decision variable in the model, just forced
    to one value), and `alpha`'s sensitivity term becomes moot (every
    feasible y is the same point). Lets the z/folding half of this MILP be
    validated in isolation against folding_ilp.py's own already-computed
    answer for the SAME external bit assignment (e.g. --force-serial
    --hard-lut-fraction 1.0 pinned to a known block_bits_*.json's values
    should reproduce that script's own Optimal/Infeasible + LUT/BRAM/cycles
    totals exactly) -- see the module's own verification notes."""
    candidate_pairs = tuple((w, a) for w in CANDIDATE_BITS for a in CANDIDATE_BITS)
    n_blocks = len(block_names)
    n_layers = len(geometries)

    # -- y[block,w,a]: sensitivity attaches here, exactly ilp_search.py's own solve_joint_bits pattern.
    y = {
        (b, w, a): pulp.LpVariable(f"y_{b}_{w}_{a}", cat=pulp.LpBinary)
        for b in block_names for w, a in candidate_pairs
    }
    raw_sensitivity = {
        (b, w, a): sensitivity[b]["sensitivity_w"][str(w)] + sensitivity[b]["sensitivity_a"][str(a)]
        for b in block_names for w, a in candidate_pairs
    }
    sens_norm = _normalize(raw_sensitivity)

    # -- z[layer,pe,simd,ram_style,w,a]: cycles/LUT/BRAM attach here, at the EXACT (fold,bits) combination.
    z: dict[tuple, pulp.LpVariable] = {}
    layer_costs: dict[tuple, dict] = {}
    raw_cycles: dict[tuple, float] = {}
    raw_lut: dict[tuple, float] = {}
    raw_bram: dict[tuple, float] = {}
    layer_folds: dict[str, list[tuple[int, int, str]]] = {}

    for layer in geometries:
        folds = _folding.candidate_folds(layer)  # respects _folding.FORCE_SERIAL if set
        layer_folds[layer.name] = folds
        for pe, simd, ram_style in folds:
            for w, a in candidate_pairs:
                cost = layer_cost_pe_simd(layer, w, a, pe, simd, ram_style)
                key = (layer.name, pe, simd, ram_style, w, a)
                layer_costs[key] = cost
                raw_cycles[key] = cost["cycles"]
                raw_lut[key] = calibrated_lut(cost["total_lut"], w, a)
                raw_bram[key] = calibrated_bram18k(cost["swu_bram18"] + cost["wm_bram18"], w, a)
                z[key] = pulp.LpVariable(f"z_{layer.name}_{pe}_{simd}_{ram_style}_{w}_{a}", cat=pulp.LpBinary)

    cycles_norm = _normalize(raw_cycles)

    prob = pulp.LpProblem("HAWQ_joint_bits_folding", pulp.LpMinimize)

    for b in block_names:
        prob += pulp.lpSum(y[(b, w, a)] for w, a in candidate_pairs) == 1, f"one_pair_per_block_{b}"

    if pinned_bits is not None:
        missing = set(block_names) - set(pinned_bits)
        if missing:
            raise ValueError(f"pinned_bits is missing block(s): {sorted(missing)}")
        for b in block_names:
            w_fixed, a_fixed = pinned_bits[b]
            if (w_fixed, a_fixed) not in candidate_pairs:
                raise ValueError(f"pinned_bits[{b!r}]=({w_fixed},{a_fixed}) not in candidate_pairs {candidate_pairs}")
            prob += y[(b, w_fixed, a_fixed)] == 1, f"pin_{b}"

    # Linking constraint -- see module docstring's worked-example explanation.
    for layer in geometries:
        folds = layer_folds[layer.name]
        for w, a in candidate_pairs:
            prob += (
                pulp.lpSum(z[(layer.name, pe, simd, ram_style, w, a)] for pe, simd, ram_style in folds)
                == y[(layer.stage, w, a)]
            ), f"link_{layer.name}_{w}_{a}"

    # Hard resource constraints -- always on, this script's headline change.
    prob += pulp.lpSum(z[k] * raw_lut[k] for k in z) <= hard_lut_fraction * XCZU7EV["LUT"], "hard_lut_budget"
    prob += pulp.lpSum(z[k] * raw_bram[k] for k in z) <= hard_bram_fraction * XCZU7EV["BRAM_18K"], "hard_bram_budget"
    if max_cycles is not None:
        prob += pulp.lpSum(z[k] * raw_cycles[k] for k in z) <= max_cycles, "max_cycles_budget"

    sensitivity_term = (1.0 / n_blocks) * pulp.lpSum(
        y[(b, w, a)] * sens_norm[(b, w, a)] for b in block_names for w, a in candidate_pairs
    )
    latency_term = (1.0 / n_layers) * pulp.lpSum(z[k] * cycles_norm[k] for k in z)
    prob += alpha * sensitivity_term + (1 - alpha) * latency_term

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap_rel))
    status_name = pulp.LpStatus[status]
    if status_name not in ("Optimal", "Infeasible"):
        raise RuntimeError(f"joint ILP hit an unexpected solver status: {status_name!r}.")

    n_binary_vars = len(y) + len(z)
    n_constraints = n_blocks + sum(len(candidate_pairs) for _ in geometries) + 2 + (1 if max_cycles is not None else 0)

    if status_name != "Optimal":
        return {
            "status": status_name,
            "alpha": alpha,
            "stage_weight_bits": {}, "stage_act_bits": {}, "per_layer": {},
            "_diagnostics": {
                "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_blocks": n_blocks, "n_layers": n_layers,
                "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
                "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
                "max_cycles": max_cycles,
                "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": _folding.FORCE_SERIAL,
                "note": f"Solver status {status_name!r} -- no joint (bits, folding) assignment satisfies the "
                        f"requested hard LUT/BRAM budget(s) (hard_lut_fraction={hard_lut_fraction}, "
                        f"hard_bram_fraction={hard_bram_fraction})"
                        + (f" and max_cycles={max_cycles:.0f}" if max_cycles is not None else "") + " at all.",
            },
        }

    stage_weight_bits: dict[str, int] = {}
    stage_act_bits: dict[str, int] = {}
    for b in block_names:
        chosen = [(w, a) for w, a in candidate_pairs if pulp.value(y[(b, w, a)]) > 0.5]
        assert len(chosen) == 1, f"block {b}: expected exactly one (w,a) pair chosen, got {chosen}"
        stage_weight_bits[b], stage_act_bits[b] = chosen[0]

    per_layer: dict[str, dict] = {}
    for layer in geometries:
        w, a = stage_weight_bits[layer.stage], stage_act_bits[layer.stage]
        folds = layer_folds[layer.name]
        chosen_fold = None
        for pe, simd, ram_style in folds:
            if pulp.value(z[(layer.name, pe, simd, ram_style, w, a)]) > 0.5:
                chosen_fold = (pe, simd, ram_style)
                break
        assert chosen_fold is not None, f"layer {layer.name}: no folding choice selected at its block's chosen (w,a)={w,a}"
        pe, simd, ram_style = chosen_fold
        cost = layer_costs[(layer.name, pe, simd, ram_style, w, a)]
        per_layer[layer.name] = {
            "stage": layer.stage, "pe": pe, "simd": simd, "ram_style": ram_style,
            "weight_bits": w, "act_bits": a, **cost,
        }

    total_lut = sum(calibrated_lut(v["total_lut"], v["weight_bits"], v["act_bits"]) for v in per_layer.values())
    total_bram = sum(
        calibrated_bram18k(v["swu_bram18"] + v["wm_bram18"], v["weight_bits"], v["act_bits"])
        for v in per_layer.values()
    )
    total_uram = sum(v.get("wm_uram18", 0) for v in per_layer.values())
    total_cycles = sum(v["cycles"] for v in per_layer.values())

    return {
        "status": status_name,
        "alpha": alpha,
        "stage_weight_bits": stage_weight_bits,
        "stage_act_bits": stage_act_bits,
        "per_layer": per_layer,
        "_diagnostics": {
            "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_blocks": n_blocks, "n_layers": n_layers,
            "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
            "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"],
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"],
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "total_uram18": total_uram, "total_cycles": total_cycles,
            "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
            "max_cycles": max_cycles,
            "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": _folding.FORCE_SERIAL,
            "note": "Joint MILP: y[block,w,a] (sensitivity) linked to z[layer,pe,simd,ram_style,w,a] "
                    "(cycles/LUT/BRAM) via a per-(layer,w,a) equality constraint -- LUT/BRAM are REAL hard "
                    "<= XCZU7EV constraints here (not a soft penalty, unlike ilp_search.py/folding_ilp.py's "
                    "own default). Objective = alpha*mean(sens_norm) + (1-alpha)*mean(cycles_norm), each "
                    "mean-normalized by its own term count (n_blocks/n_layers) so alpha is a genuine, "
                    "comparable dial -- see module docstring. GUARANTEED to fit the requested hard budget(s) "
                    "under this cost model's own calibration (still only two real-synthesis anchor points, "
                    "avg_bits=3.52/8 -- a steering signal at that calibration, not a certified hardware "
                    "guarantee, same caveat ilp_search.py/folding_ilp.py already carry).",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_12_separable_dense_relu.")
    parser.add_argument("--sensitivity-file", type=Path, required=True,
                         help="block_sensitivity_*.json (block_sensitivity.py output) or sensitivity_*.json "
                              "(sensitivity.py output, stage-level) -- must match --granularity.")
    parser.add_argument("--granularity", choices=["stage", "block"], default="block",
                         help="'block' (default) tags each layer with its owning individual bottleneck, "
                              "matching block_sensitivity.py's own per-block keys. 'stage' tags each layer "
                              "with one of ENet's 5 static stage groups, matching sensitivity.py's coarser "
                              "output. Must match --sensitivity-file's own granularity.")
    parser.add_argument("--candidate-bits", type=str, default=None,
                         help="Comma-separated override for the module-level CANDIDATE_BITS (e.g. '4,8', "
                              "this session's established 'min4' convention) -- keeps candidate_pairs small "
                              "(2x2=4 instead of the full module default (2,4,8,16)'s 4x4=16), which matters "
                              "here since z is indexed over (layer,fold,w,a): every extra (w,a) pair "
                              "multiplies the folding candidate count for EVERY layer.")
    parser.add_argument("--alpha", type=float, required=True,
                         help="Objective dial: alpha*mean(normalized sensitivity) + (1-alpha)*mean(normalized "
                              "cycles). alpha=1.0 is pure accuracy-proxy (bits chosen to minimize sensitivity "
                              "among whatever fits the hard LUT/BRAM budget); alpha=0.0 is pure latency (bits "
                              "chosen to most PERMIT low-cycle folding under that same budget -- cycles has "
                              "no direct bit-width term, so this is an indirect effect, see module docstring).")
    parser.add_argument("--hard-lut-fraction", type=float, default=1.0,
                         help="Hard `<= FRACTION * XCZU7EV['LUT']` constraint (calibrated). Always enforced -- "
                              "this script's whole point is a real hard budget, not a soft penalty. Pass e.g. "
                              "0.7 to reserve headroom for DSP/FF (resources this cost model doesn't estimate).")
    parser.add_argument("--hard-bram-fraction", type=float, default=1.0, help="Same as --hard-lut-fraction, for BRAM_18K.")
    parser.add_argument("--max-latency-ms", type=float, default=None,
                         help="Hard cap on total cycles, expressed as a latency budget at --clock-mhz (default "
                              "None = no cap). Converted to max_cycles = max_latency_ms/1000 * clock_mhz*1e6 and "
                              "added as a real <= constraint (same mechanism as --hard-lut-fraction/--hard-bram-"
                              "fraction) -- makes every alpha in [0,1] automatically respect the budget by "
                              "construction, rather than needing to sweep, solve, and discard whichever alphas "
                              "come out too slow. See solve_joint's own docstring for the full rationale.")
    parser.add_argument("--clock-mhz", type=float, default=100.0,
                         help="Clock frequency --max-latency-ms is expressed against (default 100.0, matching "
                              "this session's own established '@100MHz' reporting convention throughout).")
    parser.add_argument("--force-serial", action="store_true",
                         help="Force FOLDING_SERIAL (PE=SIMD=1) on every layer before solving -- same mechanism "
                              "as folding_ilp.py's own flag (reused directly, not reimplemented). Mainly for "
                              "validating this script against folding_ilp.py's own already-computed answers "
                              "(pin bits to a known block_bits_*.json's values externally and compare), not "
                              "part of the normal alpha-sweep workflow.")
    parser.add_argument("--time-limit", type=int, default=1800,
                         help="CBC wall-clock cap in seconds (default 1800 = 30min). This problem is ~4x "
                              "folding_ilp.py's own largest reference case (4939 candidates at min4) plus two "
                              "simultaneous binding hard constraints -- folding_ilp.py's own docstring "
                              "documents a SMALLER single-hard-constraint case taking over an hour unbounded. "
                              "Status still reports 'Optimal' if the time limit (not a genuine B&B proof) is "
                              "what stopped it -- PuLP/CBC's LpStatus doesn't distinguish the two.")
    parser.add_argument("--gap-rel", type=float, default=0.02,
                         help="CBC relative optimality gap to accept (default 0.02 = accept anything CBC can "
                              "prove is within 2%% of the true optimum).")
    parser.add_argument("--pin-bits-file", type=Path, default=None,
                         help="TEST-ONLY: a block_bits_*.json (stage_weight_bits/stage_act_bits) to pin y "
                              "to instead of letting alpha decide -- validates the z/folding half of this "
                              "MILP against folding_ilp.py's own already-computed answer for the same "
                              "external bit assignment. --alpha is still required but has no effect on the "
                              "result when this is set.")
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    if args.config != "config_23_1":
        load_config(args.config)

    if args.candidate_bits is not None:
        global CANDIDATE_BITS
        CANDIDATE_BITS = tuple(sorted(int(b) for b in args.candidate_bits.split(",")))
        print(f"Overriding CANDIDATE_BITS to {CANDIDATE_BITS} (from --candidate-bits).")

    if args.force_serial:
        _folding.FORCE_SERIAL = True
        print("--force-serial: every layer restricted to FOLDING_SERIAL (PE=SIMD=1) before solving.")

    with open(args.sensitivity_file) as f:
        sensitivity = json.load(f)

    model = ENet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, decoder_type=DECODER_TYPE,
        use_asymmetric=USE_ASYMMETRIC, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED, use_prelu=globals().get("USE_PRELU", True), prelu_variant=PRELU_VARIANT,
        use_dsc=globals().get("USE_DSC", False), dsc_no_projection=globals().get("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=globals().get("DSC_NO_PROJECTION_CONTEXT_ONLY", False),
        reg_bookend_dsc=globals().get("REG_BOOKEND_DSC", False),
        dsc_separable=globals().get("DSC_SEPARABLE", False),
    )
    if args.granularity == "block":
        geometries, block_names = dump_block_layer_geometry(model, INPUT_HW)
    else:
        geometries = dump_layer_geometry(model, INPUT_HW)
        block_names = sorted({g.stage for g in geometries})

    sensitivity_names = set(sensitivity.keys())
    missing_in_sensitivity = set(block_names) - sensitivity_names
    if missing_in_sensitivity:
        raise ValueError(
            f"--sensitivity-file is missing entries for: {sorted(missing_in_sensitivity)} -- traced model "
            f"blocks and --sensitivity-file's own top-level keys must match 1:1 (--granularity {args.granularity})."
        )
    block_names = tuple(block_names)

    candidate_pairs_count = len(CANDIDATE_BITS) ** 2
    n_z = sum(len(_folding.candidate_folds(g)) for g in geometries) * candidate_pairs_count
    n_y = len(block_names) * candidate_pairs_count
    print(f"Traced {len(geometries)} layers / {len(block_names)} blocks ({args.granularity} granularity). "
          f"candidate_bits={CANDIDATE_BITS} -> {candidate_pairs_count} (w,a) pairs. "
          f"y: {n_y} binaries, z: {n_z} binaries ({n_y + n_z} total). Solving (time_limit={args.time_limit}s, "
          f"gap_rel={args.gap_rel})...")

    pinned_bits = None
    if args.pin_bits_file is not None:
        with open(args.pin_bits_file) as f:
            pin_source = json.load(f)
        pinned_bits = {
            b: (pin_source["stage_weight_bits"][b], pin_source["stage_act_bits"][b]) for b in block_names
        }
        print(f"--pin-bits-file: y pinned to {args.pin_bits_file} for all {len(block_names)} blocks (alpha ignored).")

    max_cycles = None
    if args.max_latency_ms is not None:
        max_cycles = args.max_latency_ms / 1000 * args.clock_mhz * 1e6
        print(f"--max-latency-ms {args.max_latency_ms} @ {args.clock_mhz}MHz -> max_cycles={max_cycles:.0f} "
              f"(hard constraint).")

    result = solve_joint(
        sensitivity, geometries, block_names, args.alpha, args.hard_lut_fraction, args.hard_bram_fraction,
        args.time_limit, args.gap_rel, max_cycles=max_cycles, pinned_bits=pinned_bits,
    )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"ILP status: {result['status']}")
    if result["status"] != "Optimal":
        print(f"No joint (bits, folding) assignment satisfies the requested hard budget(s) -- see "
              f"{args.out_file}'s own _diagnostics.note.")
        return
    print(f"stage_weight_bits: {result['stage_weight_bits']}")
    print(f"stage_act_bits:    {result['stage_act_bits']}")
    diag = result["_diagnostics"]
    print(f"LUT used (calibrated): {diag['total_lut_calibrated']:.0f} ({diag['lut_pct_of_budget']:.1f}% of "
          f"{XCZU7EV['LUT']} budget) -- GUARANTEED (hard constraint).")
    print(f"BRAM_18K used (calibrated): {diag['total_bram18k_calibrated']:.0f} ({diag['bram_pct_of_budget']:.1f}% "
          f"of {XCZU7EV['BRAM_18K']} budget) -- GUARANTEED (hard constraint).")
    print(f"Total cycles (sum, ~= per-image latency): {diag['total_cycles']:.0f}")


if __name__ == "__main__":
    main()
