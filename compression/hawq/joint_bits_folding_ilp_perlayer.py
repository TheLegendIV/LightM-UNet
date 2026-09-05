"""Per-LAYER combined bits+folding MILP -- a sibling to joint_bits_folding_
ilp.py, kept completely separate and unmodified (still fully usable/
fallback-able for the per-BLOCK case -- nothing in this file touches it or
any of block_sensitivity.py/CombinedQuantENet.py/its trainers). Assigns an
independent (weight_bits, act_bits) pair PER INDIVIDUAL CONV/CONVTRANSPOSE/
MAXPOOL LAYER instead of per whole bottleneck block.

WHY THIS IS A SMALL, TARGETED DELTA, NOT A REWRITE: joint_bits_folding_ilp.py's
own z variable (folding: pe/simd/ram_style, and the cycles/LUT/BRAM cost it
carries) is ALREADY indexed per individual real FINN layer (one entry per
Conv2d/ConvTranspose2d/MaxPool2d module, from finn_block_costs.
dump_block_layer_geometry -- 104 for S12, confirmed exact key-set match
against compression/hawq/layer_sensitivity.py's own 101 conv/convtranspose
entries). It was only y (the sensitivity/bit-width-CHOICE variable) that was
block-scoped, broadcast down to every layer in a block via an equality
linking constraint. This file reindexes y by layer NAME instead of the
layer's owning block/stage and drops that broadcast -- removing an
indirection rather than adding one. The z/folding half, the hard LUT/BRAM/
max-cycles constraints, and the CBC-solve machinery are otherwise IDENTICAL
to joint_bits_folding_ilp.py's own.

SENSITIVITY SOURCE: compression/hawq/layer_sensitivity.py's own output
(layer_sensitivity_<config>.json), NOT block_sensitivity.py's -- keyed by the
identical per-conv-layer dotted names finn_block_costs.dump_block_layer_
geometry already produces. MaxPool2d layers (3 for S12: initial.pool/
down1.pool/down2.pool) have no weight tensor and were never HAWQ-measured
(layer_sensitivity.py only targets Conv2d/ConvTranspose2d) -- they get a
fixed raw sensitivity of 0.0 for every (w,a) candidate here. Because
_normalize (imported from ilp_search.py, reused unchanged) is a single
GLOBAL affine min-max map applied identically to every raw value, a constant
0.0 across all of one layer's own (w,a) candidates maps to the SAME
normalized number for all of them too -- it adds a fixed constant to the
objective, but can never influence WHICH (w,a) is optimal for that layer;
MaxPool sites are chosen purely on the (1-alpha) latency/resource term, which
is the only real information there is for them.

PREDECESSOR-CORRECTED ACT SENSITIVITY (the fix): layer_sensitivity.py
measures a layer's sensitivity from a forward hook on that SAME layer's own
module -- i.e. its OWN OUTPUT. But finn_cost_model.py's per-layer cost
formula uses a layer's `act_bits` as its OWN INPUT stream's bit-width (see
finn_cost_formulae.md's BRAM_swu term, `ceil(C*A/36)`, sizing the line
buffer that receives the INCOMING feature map). Naively using
sensitivity[L]["sensitivity_a"] to price z[L,...,a]'s accuracy term
therefore evaluates the accuracy signal on a DIFFERENT physical tensor than
the one the cost model's `a` actually represents -- L's own output, not
L's own input (= whatever fed it, several ops upstream through BN/
activation/residual-add glue that don't get their own HAWQ measurement at
all). This file corrects that: compression/hawq/layer_topology.
compute_predecessor_map traces a PLAIN FP32 mirror of the architecture via
torch.fx and returns, for every layer, the REAL upstream layer(s) whose
output actually becomes its input. `raw_sensitivity`'s act term is built
from `max(sensitivity[pred]["sensitivity_a"][a] for pred in real
predecessors)` instead of `sensitivity[L]["sensitivity_a"][a]` -- MAX
(not mean/sum) because a residual join can have TWO real predecessors (e.g.
a downsampling bottleneck's pooled branch and its own reduce/conv/expand
branch), and a wire is only as safe to compress as its most sensitive real
contributor. A layer with no real predecessor (the network's very first
tracked layer -- its input is the raw, unmeasured network input) or whose
predecessor(s) could not be traced (see KNOWN LIMITATION below) falls back
to its OWN sensitivity -- the pre-existing, imperfect convention -- since no
better signal is available.

RESOLVED LIMITATION (was real, now fixed in layer_topology.py itself):
torch.fx's plain symbolic tracer cannot follow a shape-dependent Python
conditional (`if tensor.shape[...] < ...:`), and ENet.py's own
DownsamplingBottleneck/UpsamplingBottleneck/ENet.forward's own trailing
interpolate check all have one -- compute_predecessor_map now overrides
Tracer.to_bool to sidestep every one of these at once (see its own module
docstring), so the WHOLE real S12 architecture traces with full internal
visibility, no opaque leaf modules and no per-layer fallback needed for the
Downsampling/UpsamplingBottleneck sites specifically. If tracing the whole
model ever fails outright for some other reason (a future config), this
file still logs a warning and falls back to self-sensitivity for EVERY
layer, matching this script's pre-fix behavior exactly rather than
crashing -- but that path is not expected to trigger for this architecture
anymore. See compression/hawq/demo_predecessor_fix.py for a worked,
from-scratch demonstration of the bug and the fix on a small 1-block
network.

VARIABLES (same roles as joint_bits_folding_ilp.py's own, renamed/reindexed):
    y[layer, w, a]   one-hot per INDIVIDUAL LAYER (not block) -- sensitivity
                     attaches here.
    z[layer, pe, simd, ram_style, w, a]   UNCHANGED in spirit from
                     joint_bits_folding_ilp.py -- one binary per (layer, fold,
                     w, a) combination; cycles/LUT/BRAM attach here.

LINKING CONSTRAINT -- now a same-layer identity, not a block broadcast:
    for layer, (w,a):  sum_{pe,simd,ram_style} z[layer,...,w,a] == y[layer,w,a]

OBJECTIVE -- y and z now share the SAME index cardinality (n_layers) by
construction (y is one-hot per layer here, exactly like z's own fold-marginal
already was for every layer) -- unlike the block version there is no
cardinality mismatch between the two terms to correct for, but both are
still divided by n_layers for direct alpha-convention comparability with
joint_bits_folding_ilp.py's own runs:
    sensitivity_term = (1/n_layers) * sum_{layer,w,a} y[layer,w,a] * sens_norm[layer,w,a]
    latency_term     = (1/n_layers) * sum_{layer,fold,w,a} z[layer,fold,w,a] * cycles_norm[layer,fold,w,a]
    minimize alpha * sensitivity_term + (1-alpha) * latency_term

HARD CONSTRAINTS / --force-serial: byte-identical mechanism to
joint_bits_folding_ilp.py's own (real hard `<=` LUT/BRAM/max-cycles
constraints, --force-serial reuses folding_ilp.FORCE_SERIAL the same way).

--pin-bits-file now expects a layer_bits_*.json's own
{"layer_weight_bits": {...}, "layer_act_bits": {...}} shape (one entry per
layer name) instead of a block_bits_*.json's {"stage_weight_bits": {...},
"stage_act_bits": {...}} (one entry per block name) -- same TEST-ONLY role:
validates the z/folding half of THIS script against joint_bits_folding_
ilp.py's own already-computed answer, by pinning y to a KNOWN block-uniform
assignment broadcast down to every layer (e.g. via
nnunetv2.nets.LayerQuantENet.expand_block_bits_to_layer_bits's own sibling
logic, or by hand) and comparing --force-serial --hard-lut-fraction 1.0
totals -- see this module's own __main__ verification for the exact check.

--granularity is DROPPED entirely (unlike joint_bits_folding_ilp.py's own
stage/block choice) -- since y is now indexed by individual layer name
regardless, dump_block_layer_geometry's own layer set (the one whose names
are cross-verified to match layer_sensitivity.py's) is always used; there is
no longer a coarser "stage" grouping this script has any use for on the
sensitivity axis (layer.stage is still recorded per per_layer entry purely
for traceability/reporting, same field, just no longer load-bearing for
bit-width linking).

OUTPUT SCHEMA: {"layer_weight_bits": {...}, "layer_act_bits": {...},
"per_layer": {...}, ...} -- one entry per individual layer (104 for S12,
including the 3 MaxPool2d ones), matching layer_sensitivity.py's own naming
convention.

SCOPE BOUNDARY (deliberately not solved here): this is coarser than
nnunetv2.nets.LayerQuantENet's own full per-QUANTIZER-SITE deployment schema
-- e.g. a single RegularBottleneck's reduce/conv_bn_act/residual_add/out_act
activations each get their OWN independent bit-width in LayerQuantENet,
whereas this ILP's z/y (and the underlying FINN folding cost model) assign
only ONE act_bits per whole conv "layer"'s own dataflow stream. Deploying
this ILP's output through LayerQuantENet therefore still needs a broadcast/
expansion step (each activation-only site inheriting its nearest owning
conv layer's chosen act_bits) -- a genuinely separate, later piece of work,
not needed for this file's own job (producing a correct, real per-layer
bit+folding solve) to be complete and useful on its own.

Usage:
    python compression/hawq/joint_bits_folding_ilp_perlayer.py \\
        --config config_12_separable_dense_relu \\
        --sensitivity-file compression/hawq/artifacts/layer_sensitivity_12_separable_dense_relu.json \\
        --candidate-bits 4,8 --alpha 0.3 \\
        --out-file compression/hawq/artifacts/layer_bits_folding_12_separable_dense_relu_joint_alpha0.3.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pulp
import torch  # noqa: F401 -- imported for side effect parity with joint_bits_folding_ilp.py's own model-tracing setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_23_1 import (  # noqa: E402
    BOTTLENECKS_PER_STAGE, CHANNELS, CONTEXT_PATTERN, DECODER_TYPE, IN_CHANNELS,
    OUT_CHANNELS, PRELU_VARIANT, SEPARABLE_DILATED, USE_ASYMMETRIC,
)
import folding_ilp as _folding  # noqa: E402 -- reused for candidate_folds/candidate_swu_simd/FORCE_SERIAL
from finn_block_costs import dump_block_layer_geometry  # noqa: E402
from finn_cost_model import LayerGeometry, calibrated_bram18k, calibrated_lut, layer_cost_pe_simd  # noqa: E402
from finn_stage_costs import INPUT_HW  # noqa: E402
from ilp_search import _normalize  # noqa: E402 -- one source of truth for the [0,1] normalization helper
from layer_topology import compute_predecessor_map  # noqa: E402 -- the predecessor-correction fix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624}  # same values as joint_bits_folding_ilp.py's own copy
CANDIDATE_BITS = (2, 4, 6, 8, 16)


def load_config(config_module: str) -> None:
    """Same pattern as joint_bits_folding_ilp.py's own loader."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def _act_sensitivity_sources(name: str, predecessor_map: dict[str, list[str]] | None) -> list[str]:
    """Which layer name(s) sensitivity[...]["sensitivity_a"] should be read
    from for `name`'s OWN act_bits decision -- see module docstring's
    PREDECESSOR-CORRECTED ACT SENSITIVITY section. Falls back to `name`
    itself (the pre-existing, imperfect convention) when: predecessor_map
    is None (tracing failed for the whole model), `name` has no entry in it
    (inside an opaque leaf_module_types boundary), or the entry is an empty
    list (name is the network's own first tracked layer, no real
    predecessor exists at all)."""
    if predecessor_map is None:
        return [name]
    preds = predecessor_map.get(name)
    return preds if preds else [name]


def solve_joint_perlayer(
    sensitivity: dict, geometries: list[LayerGeometry], alpha: float,
    hard_lut_fraction: float, hard_bram_fraction: float,
    time_limit: int, gap_rel: float, max_cycles: float | None = None,
    pinned_bits: dict[str, tuple[int, int]] | None = None,
    predecessor_map: dict[str, list[str]] | None = None,
) -> dict:
    """The per-layer combined MILP -- see module docstring for the full
    formulation and how it differs from joint_bits_folding_ilp.solve_joint
    (y reindexed from block to individual layer name; everything else --
    z's own construction, the hard resource constraints, max_cycles,
    pinned_bits's role -- is the same mechanism, just keyed by layer.name
    throughout instead of layer.stage).

    predecessor_map: from layer_topology.compute_predecessor_map (or None to
    skip the correction entirely, reproducing this file's original
    self-indexed behavior byte-for-byte) -- see module docstring's
    PREDECESSOR-CORRECTED ACT SENSITIVITY section for what this fixes and
    _act_sensitivity_sources for the exact fallback rules."""
    candidate_pairs = tuple((w, a) for w in CANDIDATE_BITS for a in CANDIDATE_BITS)
    layer_names = tuple(g.name for g in geometries)
    n_layers = len(geometries)

    # -- y[layer,w,a]: sensitivity attaches here. Layers absent from
    # `sensitivity` (the 3 MaxPool2d sites for S12 -- no weight, never HAWQ-
    # measured) get a constant raw sensitivity of 0.0 for every (w,a) --
    # see module docstring for why this can bias the objective's constant
    # offset but never which (w,a) is chosen for that specific layer.
    y = {
        (name, w, a): pulp.LpVariable(f"y_{name}_{w}_{a}", cat=pulp.LpBinary)
        for name in layer_names for w, a in candidate_pairs
    }
    raw_sensitivity: dict[tuple[str, int, int], float] = {}
    for name in layer_names:
        act_sources = [s for s in _act_sensitivity_sources(name, predecessor_map) if s in sensitivity]
        for w, a in candidate_pairs:
            sens_w = sensitivity[name]["sensitivity_w"][str(w)] if name in sensitivity else 0.0
            # MAX across real predecessor(s), not self -- see module docstring.
            # Falls back to 0.0 only if NEITHER name nor any of its real
            # predecessors have a sensitivity entry at all (e.g. name is a
            # MaxPool2d whose own predecessor is ALSO a MaxPool2d, both
            # unmeasured -- rare, but a real possibility worth a clean
            # fallback rather than a KeyError).
            sens_a = max((sensitivity[s]["sensitivity_a"][str(a)] for s in act_sources), default=0.0)
            raw_sensitivity[(name, w, a)] = sens_w + sens_a
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

    prob = pulp.LpProblem("HAWQ_joint_bits_folding_perlayer", pulp.LpMinimize)

    for name in layer_names:
        prob += pulp.lpSum(y[(name, w, a)] for w, a in candidate_pairs) == 1, f"one_pair_per_layer_{name}"

    if pinned_bits is not None:
        missing = set(layer_names) - set(pinned_bits)
        if missing:
            raise ValueError(f"pinned_bits is missing layer(s): {sorted(missing)}")
        for name in layer_names:
            w_fixed, a_fixed = pinned_bits[name]
            if (w_fixed, a_fixed) not in candidate_pairs:
                raise ValueError(f"pinned_bits[{name!r}]=({w_fixed},{a_fixed}) not in candidate_pairs {candidate_pairs}")
            prob += y[(name, w_fixed, a_fixed)] == 1, f"pin_{name}"

    # Linking constraint -- now a same-layer identity (see module docstring).
    for layer in geometries:
        folds = layer_folds[layer.name]
        for w, a in candidate_pairs:
            prob += (
                pulp.lpSum(z[(layer.name, pe, simd, ram_style, w, a)] for pe, simd, ram_style in folds)
                == y[(layer.name, w, a)]
            ), f"link_{layer.name}_{w}_{a}"

    # Hard resource constraints -- identical mechanism to joint_bits_folding_ilp.py's own.
    prob += pulp.lpSum(z[k] * raw_lut[k] for k in z) <= hard_lut_fraction * XCZU7EV["LUT"], "hard_lut_budget"
    prob += pulp.lpSum(z[k] * raw_bram[k] for k in z) <= hard_bram_fraction * XCZU7EV["BRAM_18K"], "hard_bram_budget"
    if max_cycles is not None:
        prob += pulp.lpSum(z[k] * raw_cycles[k] for k in z) <= max_cycles, "max_cycles_budget"

    sensitivity_term = (1.0 / n_layers) * pulp.lpSum(
        y[(name, w, a)] * sens_norm[(name, w, a)] for name in layer_names for w, a in candidate_pairs
    )
    latency_term = (1.0 / n_layers) * pulp.lpSum(z[k] * cycles_norm[k] for k in z)
    prob += alpha * sensitivity_term + (1 - alpha) * latency_term

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap_rel))
    status_name = pulp.LpStatus[status]
    if status_name not in ("Optimal", "Infeasible"):
        raise RuntimeError(f"joint per-layer ILP hit an unexpected solver status: {status_name!r}.")

    n_binary_vars = len(y) + len(z)
    n_constraints = n_layers + sum(len(candidate_pairs) for _ in geometries) + 2 + (1 if max_cycles is not None else 0)

    if status_name != "Optimal":
        return {
            "status": status_name,
            "alpha": alpha,
            "layer_weight_bits": {}, "layer_act_bits": {}, "per_layer": {},
            "_diagnostics": {
                "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_layers": n_layers,
                "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
                "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
                "max_cycles": max_cycles,
                "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": _folding.FORCE_SERIAL,
                "note": f"Solver status {status_name!r} -- no joint per-layer (bits, folding) assignment "
                        f"satisfies the requested hard LUT/BRAM budget(s) (hard_lut_fraction={hard_lut_fraction}, "
                        f"hard_bram_fraction={hard_bram_fraction})"
                        + (f" and max_cycles={max_cycles:.0f}" if max_cycles is not None else "") + " at all.",
            },
        }

    layer_weight_bits: dict[str, int] = {}
    layer_act_bits: dict[str, int] = {}
    for name in layer_names:
        chosen = [(w, a) for w, a in candidate_pairs if pulp.value(y[(name, w, a)]) > 0.5]
        assert len(chosen) == 1, f"layer {name}: expected exactly one (w,a) pair chosen, got {chosen}"
        layer_weight_bits[name], layer_act_bits[name] = chosen[0]

    per_layer: dict[str, dict] = {}
    for layer in geometries:
        w, a = layer_weight_bits[layer.name], layer_act_bits[layer.name]
        folds = layer_folds[layer.name]
        chosen_fold = None
        for pe, simd, ram_style in folds:
            if pulp.value(z[(layer.name, pe, simd, ram_style, w, a)]) > 0.5:
                chosen_fold = (pe, simd, ram_style)
                break
        assert chosen_fold is not None, f"layer {layer.name}: no folding choice selected at its own chosen (w,a)={w,a}"
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
        "layer_weight_bits": layer_weight_bits,
        "layer_act_bits": layer_act_bits,
        "per_layer": per_layer,
        "_diagnostics": {
            "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_layers": n_layers,
            "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
            "total_lut_calibrated": total_lut, "xczu7ev_lut_budget": XCZU7EV["LUT"],
            "lut_pct_of_budget": 100 * total_lut / XCZU7EV["LUT"],
            "total_bram18k_calibrated": total_bram, "xczu7ev_bram18k_budget": XCZU7EV["BRAM_18K"],
            "bram_pct_of_budget": 100 * total_bram / XCZU7EV["BRAM_18K"],
            "total_uram18": total_uram, "total_cycles": total_cycles,
            "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
            "max_cycles": max_cycles,
            "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": _folding.FORCE_SERIAL,
            "note": "Joint per-LAYER MILP: y[layer,w,a] (sensitivity) linked to z[layer,pe,simd,ram_style,w,a] "
                    "(cycles/LUT/BRAM) via a same-layer equality constraint -- LUT/BRAM are REAL hard "
                    "<= XCZU7EV constraints here (not a soft penalty), same as joint_bits_folding_ilp.py's own. "
                    "Objective = alpha*mean(sens_norm) + (1-alpha)*mean(cycles_norm), both mean-normalized by "
                    "the SAME n_layers (y and z share the same index cardinality here, unlike the per-block "
                    "version). GUARANTEED to fit the requested hard budget(s) under this cost model's own "
                    "calibration (still only two real-synthesis anchor points, avg_bits=3.52/8 -- a steering "
                    "signal at that calibration, not a certified hardware guarantee, same caveat "
                    "joint_bits_folding_ilp.py already carries). Coarser than nnunetv2.nets.LayerQuantENet's "
                    "own full per-quantizer-site deployment schema -- see module docstring's SCOPE BOUNDARY.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_23_1",
                         help="Which compression/hawq/config_*.py to load -- e.g. config_12_separable_dense_relu.")
    parser.add_argument("--sensitivity-file", type=Path, required=True,
                         help="layer_sensitivity_*.json (layer_sensitivity.py output) -- one entry per "
                              "individual Conv2d/ConvTranspose2d layer name.")
    parser.add_argument("--candidate-bits", type=str, default=None,
                         help="Comma-separated override for the module-level CANDIDATE_BITS (e.g. '4,8', "
                              "this session's established 'min4' convention) -- keeps candidate_pairs small, "
                              "which matters here since z is indexed over (layer,fold,w,a): every extra (w,a) "
                              "pair multiplies the folding candidate count for EVERY layer.")
    parser.add_argument("--alpha", type=float, required=True,
                         help="Objective dial: alpha*mean(normalized sensitivity) + (1-alpha)*mean(normalized "
                              "cycles). alpha=1.0 is pure accuracy-proxy; alpha=0.0 is pure latency. See module "
                              "docstring for MaxPool2d layers' fixed zero-sensitivity handling.")
    parser.add_argument("--hard-lut-fraction", type=float, default=1.0,
                         help="Hard `<= FRACTION * XCZU7EV['LUT']` constraint (calibrated). Always enforced.")
    parser.add_argument("--hard-bram-fraction", type=float, default=1.0, help="Same as --hard-lut-fraction, for BRAM_18K.")
    parser.add_argument("--max-latency-ms", type=float, default=None,
                         help="Hard cap on total cycles, expressed as a latency budget at --clock-mhz (default "
                              "None = no cap). Converted to max_cycles = max_latency_ms/1000 * clock_mhz*1e6 and "
                              "added as a real <= constraint -- same mechanism as joint_bits_folding_ilp.py's own.")
    parser.add_argument("--clock-mhz", type=float, default=100.0,
                         help="Clock frequency --max-latency-ms is expressed against (default 100.0).")
    parser.add_argument("--force-serial", action="store_true",
                         help="Force FOLDING_SERIAL (PE=SIMD=1) on every layer before solving -- same mechanism "
                              "as folding_ilp.py's/joint_bits_folding_ilp.py's own flag (reused directly). Mainly "
                              "for validating this script against joint_bits_folding_ilp.py's own already-"
                              "computed answers (pin bits externally and compare) -- see module docstring.")
    parser.add_argument("--time-limit", type=int, default=1800,
                         help="CBC wall-clock cap in seconds (default 1800 = 30min).")
    parser.add_argument("--gap-rel", type=float, default=0.02,
                         help="CBC relative optimality gap to accept (default 0.02 = accept anything CBC can "
                              "prove is within 2%% of the true optimum).")
    parser.add_argument("--pin-bits-file", type=Path, default=None,
                         help="TEST-ONLY: a layer_bits_*.json ({'layer_weight_bits': {...}, 'layer_act_bits': "
                              "{...}}) to pin y to instead of letting alpha decide -- validates the z/folding "
                              "half of this MILP against joint_bits_folding_ilp.py's own already-computed "
                              "answer for an equivalent (block-uniform, broadcast to every layer) external bit "
                              "assignment. --alpha is still required but has no effect on the result when set.")
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
    geometries, _block_names = dump_block_layer_geometry(model, INPUT_HW)
    layer_names = tuple(g.name for g in geometries)

    # Predecessor-corrected act sensitivity -- see module docstring's
    # PREDECESSOR-CORRECTED ACT SENSITIVITY section. Never fatal: any
    # tracing failure falls back to this file's original self-indexed
    # behavior for EVERY layer, rather than crashing the whole run (not
    # expected to trigger for this architecture -- see RESOLVED LIMITATION).
    try:
        predecessor_map = compute_predecessor_map(model)
        n_resolved = sum(1 for name in layer_names if predecessor_map.get(name))
        print(f"Predecessor map: {n_resolved}/{len(layer_names)} layers have a real, traced predecessor "
              f"(the rest are the network's own first layer(s), with no real predecessor to fall back to "
              f"anything but self-sensitivity).")
    except Exception as error:
        predecessor_map = None
        print(f"WARNING: could not compute a predecessor map ({type(error).__name__}: {error}) -- falling back "
              f"to self-indexed act sensitivity for EVERY layer (this file's original, pre-fix behavior).")

    maxpool_names = {g.name for g in geometries if g.op_type == "MaxPool2d"}
    sensitivity_names = set(sensitivity.keys())
    missing_in_sensitivity = set(layer_names) - sensitivity_names
    unexpected_missing = missing_in_sensitivity - maxpool_names
    if unexpected_missing:
        raise ValueError(
            f"--sensitivity-file is missing entries for: {sorted(unexpected_missing)} -- traced model layers "
            f"and --sensitivity-file's own top-level keys must match 1:1 for every Conv2d/ConvTranspose2d layer "
            f"(only MaxPool2d layers are allowed to be absent, see module docstring)."
        )
    if missing_in_sensitivity:
        print(f"Note: {sorted(missing_in_sensitivity)} have no sensitivity entry (MaxPool2d, no weight, never "
              f"HAWQ-measured) -- given a fixed raw sensitivity of 0.0 for every (w,a), see module docstring.")

    candidate_pairs_count = len(CANDIDATE_BITS) ** 2
    n_z = sum(len(_folding.candidate_folds(g)) for g in geometries) * candidate_pairs_count
    n_y = len(layer_names) * candidate_pairs_count
    print(f"Traced {len(geometries)} layers (per-layer granularity, no block grouping). "
          f"candidate_bits={CANDIDATE_BITS} -> {candidate_pairs_count} (w,a) pairs. "
          f"y: {n_y} binaries, z: {n_z} binaries ({n_y + n_z} total). Solving (time_limit={args.time_limit}s, "
          f"gap_rel={args.gap_rel})...")

    pinned_bits = None
    if args.pin_bits_file is not None:
        with open(args.pin_bits_file) as f:
            pin_source = json.load(f)
        pinned_bits = {
            name: (pin_source["layer_weight_bits"][name], pin_source["layer_act_bits"][name]) for name in layer_names
        }
        print(f"--pin-bits-file: y pinned to {args.pin_bits_file} for all {len(layer_names)} layers (alpha ignored).")

    max_cycles = None
    if args.max_latency_ms is not None:
        max_cycles = args.max_latency_ms / 1000 * args.clock_mhz * 1e6
        print(f"--max-latency-ms {args.max_latency_ms} @ {args.clock_mhz}MHz -> max_cycles={max_cycles:.0f} "
              f"(hard constraint).")

    result = solve_joint_perlayer(
        sensitivity, geometries, args.alpha, args.hard_lut_fraction, args.hard_bram_fraction,
        args.time_limit, args.gap_rel, max_cycles=max_cycles, pinned_bits=pinned_bits,
        predecessor_map=predecessor_map,
    )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_file}")
    print(f"ILP status: {result['status']}")
    if result["status"] != "Optimal":
        print(f"No joint per-layer (bits, folding) assignment satisfies the requested hard budget(s) -- see "
              f"{args.out_file}'s own _diagnostics.note.")
        return
    diag = result["_diagnostics"]
    print(f"LUT used (calibrated): {diag['total_lut_calibrated']:.0f} ({diag['lut_pct_of_budget']:.1f}% of "
          f"{XCZU7EV['LUT']} budget) -- GUARANTEED (hard constraint).")
    print(f"BRAM_18K used (calibrated): {diag['total_bram18k_calibrated']:.0f} ({diag['bram_pct_of_budget']:.1f}% "
          f"of {XCZU7EV['BRAM_18K']} budget) -- GUARANTEED (hard constraint).")
    print(f"Total cycles (sum, ~= per-image latency): {diag['total_cycles']:.0f}")


if __name__ == "__main__":
    main()
