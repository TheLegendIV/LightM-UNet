"""Per-LAYER combined bits+folding MILP for FINN dataflow deployment.
Assigns an independent (weight_bits, act_bits) pair AND an independent
(PE, SIMD, ram_style) folding choice to every individual real FINN layer
(Conv2d/ConvTranspose2d/MaxPool2d) of an S12-family ENet, subject to hard
LUT/BRAM_18K/DSP budgets on the target device (xczu7ev), by solving one
mixed-integer program with CBC (via pulp).

2026-09-17 REFACTOR: this file used to be joint_bits_folding_ilp_perlayer.py,
a thin per-layer reindexing of an older per-BLOCK MILP (joint_bits_folding_
ilp.py) and depended on four other modules (folding_ilp.py, ilp_search.py,
finn_block_costs.py, finn_stage_costs.py) plus a stale default config
(config_23_1.py) purely for a handful of small, unchanging helpers. All of
that is now INLINED below (candidate_folds/FORCE_SERIAL, _normalize,
trace_layer_geometry/INPUT_HW) -- this is the only file needed to run the
per-layer ILP, on top of finn_cost_model.py (the cost formulae),
block_utils.py (model structural introspection), layer_topology.py (the
predecessor-correction fix, see below) and one of the S12 config_*.py files.
The old per-BLOCK files (folding_ilp.py, joint_bits_folding_ilp.py,
ilp_search.py, block_sensitivity.py, finn_block_costs.py, finn_stage_costs.py)
and non-S12 configs have been moved to archive/ -- see that directory's own
notes for what still depends on them (mainly older, non-per-layer SLURM
jobs and analysis plots for architectures other than S12).

WHY PER-LAYER, NOT PER-BLOCK: a whole bottleneck block sharing one
(weight_bits, act_bits) pair wastes accuracy headroom on its least-sensitive
layer and wastes resource budget on its most-sensitive one. Reindexing the
bit-choice variable from block to individual layer name removes that
coupling; the folding variable (z) was already indexed per individual real
FINN layer (one entry per Conv2d/ConvTranspose2d/MaxPool2d module, from
trace_layer_geometry below).

SENSITIVITY SOURCE: layer_sensitivity.py's own output (layer_sensitivity_
<config>.json), keyed by the identical per-conv-layer dotted names
trace_layer_geometry produces. MaxPool2d layers (3 for S12: initial.pool/
down1.pool/down2.pool) have no weight tensor and were never HAWQ-measured
-- they get a fixed raw sensitivity of 0.0 for every (w,a) candidate here.
Because _normalize is a single GLOBAL affine min-max map applied
identically to every raw value, a constant 0.0 across all of one layer's
own (w,a) candidates maps to the SAME normalized number for all of them
too -- it adds a fixed constant to the objective, but can never influence
WHICH (w,a) is optimal for that layer; MaxPool sites are chosen purely on
the (1-alpha) latency/resource term, which is the only real information
there is for them.

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
all). This file corrects that: layer_topology.compute_predecessor_map
traces a PLAIN FP32 mirror of the architecture via torch.fx and returns,
for every layer, the REAL upstream layer(s) whose output actually becomes
its input. `raw_sensitivity`'s act term is built from
`max(sensitivity[pred]["sensitivity_a"][a] for pred in real predecessors)`
instead of `sensitivity[L]["sensitivity_a"][a]` -- MAX (not mean/sum)
because a residual join can have TWO real predecessors (e.g. a
downsampling bottleneck's pooled branch and its own reduce/conv/expand
branch), and a wire is only as safe to compress as its most sensitive real
contributor. A layer with no real predecessor (the network's very first
tracked layer -- its input is the raw, unmeasured network input) or whose
predecessor(s) could not be traced falls back to its OWN sensitivity -- the
pre-existing, imperfect convention -- since no better signal is available.

VARIABLES:
    y[layer, w, a]   one-hot per INDIVIDUAL LAYER -- sensitivity attaches here.
    z[layer, pe, simd, ram_style, variant, w, a]   one binary per (layer,
                     fold, variant, w, a) combination; cycles/LUT/BRAM/DSP
                     attach here. `variant` (2026-09-17 addition, additive/
                     opt-in via --allow-lut-mult, see VARIANT_RTL_DSP_NOACT1/
                     VARIANT_HLS_LUT_NOACT0/candidate_folds below) is always
                     VARIANT_RTL_DSP_NOACT1 unless --allow-lut-mult is set,
                     in which case VARIANT_HLS_LUT_NOACT0 (HLS backend,
                     LUT-based multiplication instead of DSP, fused
                     activation instead of a separate Thresholding node) also
                     becomes eligible per layer -- lets the ILP spend spare
                     LUT budget instead of DSP budget on specific layers.

LINKING CONSTRAINT (same-layer identity):
    for layer, (w,a):  sum_{pe,simd,ram_style} z[layer,...,w,a] == y[layer,w,a]

OBJECTIVE -- y and z share the SAME index cardinality (n_layers) by
construction (y is one-hot per layer here, exactly like z's own fold-marginal
already is for every layer), both mean-normalized by n_layers:
    sensitivity_term = (1/n_layers) * sum_{layer,w,a} y[layer,w,a] * sens_norm[layer,w,a]
    latency_term     = (1/n_layers) * sum_{layer,fold,w,a} z[layer,fold,w,a] * cycles_norm[layer,fold,w,a]
    minimize alpha * sensitivity_term + (1-alpha) * latency_term

HARD CONSTRAINTS: real `<=` LUT/BRAM_18K/DSP/URAM/max-cycles constraints
(not a soft penalty) -- --hard-lut-fraction/--hard-bram-fraction/--hard-dsp-
fraction/--hard-uram-fraction (all default 1.0, always enforced) cap each
resource at that fraction of XCZU7EV's nominal budget. BRAM_18K includes the
standalone Thresholding_rtl's own memory (thr_bram18, noActivation=1 regime
-- see finn_cost_model.conv_cost_pe_simd) alongside the SWU line buffer and
weight memory. DSP was added 2026-09-17: uncapped, the RTL DSP-packing rule
(ceil(PE/lanes)*SIMD) let an S12-dense alpha=0.25 solve pick a plan needing
2564 DSP48s on a 1728-DSP device -- it is now always a real constraint, not
just tracked in diagnostics.

URAM went through a whole rise-and-fall arc over 2026-09-17/18, ending with
it permanently disabled for every consumer -- kept here for the record.
Added 2026-09-17: uncapped, an S12-dense alpha=0.25 solve picked wm_uram18=
205 (all on MVAU's own weight memory), but the real 8-way hardware build for
that exact solve landed at URAM=2/96 -- real hardware essentially never
realized the ILP's own "ultra" choice for MVAU weights. Response #1: MVAU's
`ram_style` hard-fixed to "block" permanently. The freed decision variable
was then repointed to SWU's own line-buffer memory, then to the standalone
Thresholding_rtl node's own memory (real data showed THAT was the dominant
real BRAM consumer, up to 24 blocks/node, while URAM sat mostly idle) -- but
both of those turned out to have the same problem MVAU did, for different
reasons: 48/48 real SWU nodes across both real hardware datasets used
ram_style="distributed" regardless of what was requested (zero real
block/ultra SWU evidence ever existed), and a REAL Vivado synthesis attempt
of Thresholding_rtl with ram_style="ultra" FAILS outright -- URAM288 cannot
be used as ROM, and Thresholding's memory is exactly that: a compile-time-
constant lookup table, never written at runtime. See finn_milp.py's own
RAM_STYLES/candidate_folds docstrings for the full blow-by-blow. Net result:
total_uram18 is now structurally 0 for every layer, always -- URAM was never
a real option on this device (xczu7ev, UltraScale+) for any consumer this
cost model has. XCZU7EV["URAM"]/--hard-uram-fraction are kept, not removed,
purely for provenance and in case a genuinely URAM-eligible consumer is ever
added later.

The freed `ram_style` decision variable was NOT left permanently inert,
though: since 2026-09-18 it drives a different, real tradeoff for the same
Thresholding_rtl node -- block (BRAM) vs. distributed (LUTRAM), not vs.
ultra. Unlike "ultra", "distributed" IS structurally synthesis-safe here
(thresholding.sv's own RAM_STYLE localparam genuinely supports it, forceable
per-node via depth_trigger_bram) -- it's just never been exercised in real
hardware (0/240 real Thresholding_rtl nodes across both datasets ever showed
nonzero real_LUTRAM, since every real node used "auto", and Vivado's own
choice always landed on BRAM for the geometries tested). So BRAM_18K usage
now includes thr_bram18 only when a layer's Thresholding memory is on
"block"; when it's on "distributed" instead, that same memory shows up in
the LUT budget via thr_lut instead, using a DERIVED (not fit) LUTRAM cost --
see finn_cost_model.py's _THR_RTL_LUTRAM_PER_PE_NUMSTEP docstring. FIFOs are
a separate resource this cost model does NOT model at all yet (see finn_cost_
model.py's own top-of-file "Still not covered" list).

--pin-bits-file expects a layer_bits_*.json's own {"layer_weight_bits":
{...}, "layer_act_bits": {...}} shape (one entry per layer name) -- TEST-
ONLY, pins y to a known assignment and skips using alpha to choose it
(--alpha is still required but has no effect on the result).

OUTPUT SCHEMA: {"layer_weight_bits": {...}, "layer_act_bits": {...},
"per_layer": {...}, "_diagnostics": {...}} -- one entry per individual
layer (88 for the S12 dense family, including the 3 MaxPool2d ones),
matching layer_sensitivity.py's own naming convention.

SCOPE BOUNDARY (deliberately not solved here): this is coarser than
nnunetv2.nets.LayerQuantENet's own full per-QUANTIZER-SITE deployment
schema -- e.g. a single RegularBottleneck's reduce/conv_bn_act/
residual_add/out_act activations each get their OWN independent bit-width
in LayerQuantENet, whereas this ILP's z/y (and the underlying FINN folding
cost model) assign only ONE act_bits per whole conv "layer"'s own dataflow
stream. expand_layer_bits.py is the separate, later bridge that expands
this file's output into that full per-site schema for actual deployment.

Usage:
    python MILP/finn_milp.py \\
        --config config_12_dense_relu_warmstart150ep \\
        --sensitivity-file MILP/artifacts/layer_sensitivity_12_dense_relu_warmstart150ep.json \\
        --candidate-bits 4,6,8 --alpha 0.25 \\
        --hard-lut-fraction 0.7 --force-dsp \\
        --out-file MILP/artifacts/layer_bits_folding_12_dense_relu_warmstart150ep_joint_alpha0.25.json
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import pulp
import torch  # noqa: F401 -- imported for side-effect parity with the model-tracing setup below

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_utils import enumerate_blocks, path_to_block_map  # noqa: E402
from finn_cost_model import (  # noqa: E402
    IMPL_STYLE_HLS, IMPL_STYLE_RTL, RAM_STYLE_BLOCK, RAM_STYLE_ULTRA, LayerGeometry, calibrated_bram18k,
    calibrated_lut, divisors, layer_cost_pe_simd, max_pe, max_simd,
)
from layer_topology import compute_predecessor_map  # noqa: E402 -- the predecessor-correction fix

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.ENet import ENet  # noqa: E402

XCZU7EV = {"LUT": 230_400, "BRAM_18K": 624, "DSP": 1_728, "URAM": 96}  # xczu7ev-ffvc1156-2-e (DSP48E2 count matches hardware/results.csv's DSP_pct column; URAM=96 real URAM288 blocks, 96*288Kib=27Mib, confirmed against the real S12-dense-warmstart hardware build's own URAM=2/96 utilization row)
CANDIDATE_BITS = (2, 4, 6, 8, 16)
INPUT_HW = (512, 512)  # real nnU-Net patch size (see debug.json's configuration_manager.patch_size)
RAM_STYLES = (RAM_STYLE_BLOCK, "distributed")  # 2026-09-18: a free choice again, FOURTH state in two
# days, but a different axis than any before -- block vs. URAM ("ultra") for MVAU weight memory, then
# SWU's line buffer, then Thresholding's own memory, each in turn found to have NO real, working URAM
# path on this device (xczu7ev, UltraScale+): real hardware never materialized MVAU's "ultra" request,
# 48/48 real SWU nodes always used "distributed" regardless of what was requested, and a real Vivado
# synthesis attempt of Thresholding_rtl with ram_style="ultra" FAILS outright (URAM288 cannot be used
# as ROM -- Thresholding's memory is a compile-time-constant lookup table, never written at runtime,
# and URAM lacks the INIT-file-based ROM initialization mechanism BRAM primitives have). URAM is now
# permanently 0 for every layer and every consumer -- XCZU7EV["URAM"]/--hard-uram-fraction are kept,
# not removed, purely for provenance and in case a genuinely URAM-eligible consumer is ever added.
#
# This axis now drives block-vs-DISTRIBUTED for Thresholding_rtl's own memory instead (thr_ram_style)
# -- a real LUT-vs-BRAM tradeoff, not a LUT-vs-URAM one. Unlike "ultra", "distributed" IS structurally
# real here: thresholding.sv's own RAM_STYLE localparam has a genuine "auto"/"distributed" branch
# (confirmed via direct .sv source read), forceable per-node via depth_trigger_bram (see thresholding.sv
# comment inline where DEPTH_TRIGGER_BRAM resolves to "distributed" when the node's own depth never
# reaches that trigger). It's just never been exercised in any real build -- every real node so far used
# "auto" (i.e. Vivado's own choice), which always landed on BRAM for the geometries tested (0/240 real
# Thresholding_rtl nodes across both datasets ever showed nonzero real_LUTRAM). The LUTRAM cost
# (_THR_RTL_LUTRAM_PER_PE_NUMSTEP in finn_cost_model.py) is DERIVED the same way the now-dead URAM
# constant was -- scaled from the real, validated BRAM18 constant by the real capacity ratio to FINN's
# own LUTRAM primitive shape (finn.util.basic.mem_primitives_versal: "LUTRAM": (1, 64), a real Xilinx
# RAM64X1S shape) -- so it correctly predicts distributed is a BAD deal at this project's typical
# numSteps (e.g. ~840 LUTs vs ~3 BRAM18 blocks at numSteps=255), matching why Vivado's own "auto" never
# picks it; it only wins at small numSteps, where BRAM's own fixed per-block overhead dominates -- a
# real per-layer resource-pressure tradeoff an ILP can evaluate, unlike a fixed local heuristic.
FORCE_SERIAL = False  # set True by --force-serial: restricts every layer to (PE, SIMD) = (1, 1)

# Curated per-layer resource variants (2026-09-17 addition, additive/opt-in
# via --allow-lut-mult -- see module docstring's RESOURCE VARIANTS section).
# "rtl_dsp_noact1" is today's ONLY variant, unchanged in every existing
# invocation. "hls_lut_noact0" is the new "spend spare LUT instead of DSP"
# option: HLS backend, LUT-based multiplication (force_dsp=False), fused
# activation (no separate Thresholding node). Deliberately NOT a full
# {hls,rtl}x{dsp,lut}x{0,1} cross product -- rtl_dsp_noact0 is never a legal
# FINN config (MVAU_rtl structurally requires noActivation=1), and
# hls_dsp_noact1 is dominated by rtl_dsp_noact1 wherever RTL is legal (same
# DSP pricing, without RTL's own derating/packing efficiency) so it's left
# out of the curated set for now -- see the plan this was built from,
# C:\Users\win32\.claude\plans\the-current-ilp-inherited-lynx.md.
VARIANT_RTL_DSP_NOACT1 = "rtl_dsp_noact1"
VARIANT_HLS_LUT_NOACT0 = "hls_lut_noact0"

# VARIANT_HLS_DSP_NOACT0 -- PLACEHOLDER ONLY, not yet wired into
# candidate_folds/any CLI flag (deliberately absent from every variants list
# below, so its mere existence here has ZERO effect on any run). Real FINN
# config: HLS backend, DSP-based multiplication (force_dsp=True), fused
# activation (no separate Thresholding node) -- i.e. "keep DSP, but still
# shed the standalone Thresholding node's cost." _variant_cost_kwargs
# already handles it correctly (same 3 already-implemented cost-model knobs,
# no new formula needed), but do NOT add it to any layer's eligible variant
# list without ALSO hard-restricting that layer's own candidate folds to
# PE<=SIMD first -- confirmed via hardware/datasets/mvau_lut_calibration_
# dataset.csv (89 real MVAU_hls nodes, ALL resType=dsp/force_dsp=True,
# fused-threshold outputDataType -- i.e. exactly this variant) and hardware/
# mvau_lut_correlation_report.txt: this repo's own structural LUT formula
# (mult_luts+addertree_luts+acc_luts, the same family thr_luts_fused/
# comp_luts_fused extends) gets R^2=0.009 against that real data -- "beyond
# useless" -- while PE alone correlates at 0.69 and SIMD is NEGATIVELY
# correlated (-0.41); PE>SIMD rows are 36% of that dataset but 70.7% of its
# real_LUT (see --require-simd-ge-pe's own docstring, same finding). This is
# NOT evidence against VARIANT_HLS_LUT_NOACT0 above (that dataset is 100%
# resType=dsp, zero variance, so it says nothing about resType=lut behavior)
# -- it is specifically about the DSP-multiplier + fused-threshold
# combination this placeholder represents. When this variant is actually
# enabled, bake PE<=SIMD into ITS OWN eligibility in candidate_folds (not a
# bolt-on --require-simd-ge-pe the user has to remember to pass).
VARIANT_HLS_DSP_NOACT0 = "hls_dsp_noact0"

ALLOW_LUT_MULT = False  # set True by --allow-lut-mult: makes VARIANT_HLS_LUT_NOACT0 eligible alongside the RTL default


def _variant_cost_kwargs(variant: str, force_dsp: bool) -> dict:
    """Maps a curated `variant` to the (impl_style, force_dsp, no_activation)
    triple conv_cost_pe_simd (via layer_cost_pe_simd's **kw) expects.
    "rtl_dsp_noact1" honors the CLI-level --force-dsp flag exactly as every
    existing caller already does (in practice always DSP regardless of the
    flag, since impl_style=rtl alone already forces it -- see finn_cost_
    model.py's conv_cost_pe_simd docstring). "hls_lut_noact0" is DSP-free by
    construction -- --force-dsp has no effect on it, deliberately, since
    forcing DSP on the free-LUT variant would defeat its whole purpose.
    "hls_dsp_noact0" is a PLACEHOLDER (see its own module-level comment
    above) -- handled here for completeness/testability, but never reachable
    from any live run since it's never added to a layer's variants list."""
    if variant == VARIANT_RTL_DSP_NOACT1:
        return {"impl_style": IMPL_STYLE_RTL, "force_dsp": force_dsp, "no_activation": True}
    if variant == VARIANT_HLS_LUT_NOACT0:
        return {"impl_style": IMPL_STYLE_HLS, "force_dsp": False, "no_activation": False}
    if variant == VARIANT_HLS_DSP_NOACT0:
        return {"impl_style": IMPL_STYLE_HLS, "force_dsp": True, "no_activation": False}
    raise ValueError(f"unknown resource variant {variant!r}")


def _calibration_force_dsp(variant_kwargs: dict) -> bool:
    """The `force_dsp` calibrated_lut/calibrated_bram18k should be told for
    a given variant's own cost -- NOT simply variant_kwargs["force_dsp"]
    (the raw CLI flag value), because that flag is not the same question as
    "was this node's multiplication actually forced onto DSP". Mirrors
    conv_cost_pe_simd's own `use_dsp = force_dsp or impl_style == IMPL_STYLE_
    RTL` exactly: rtl_dsp_noact1 ALWAYS gets _RTL_MVU_LUT_DERATE applied
    inside conv_cost_pe_simd regardless of --force-dsp (impl_style=RTL alone
    forces it), so calibrated_lut must ALWAYS bypass its own avg_bits table
    for this variant too -- not just when --force-dsp happens to be set.
    Every run so far has passed --force-dsp, which made variant_kwargs
    ["force_dsp"] already True for rtl_dsp_noact1 and masked this: a run
    WITHOUT --force-dsp would otherwise double/mis-derate RTL's already-
    node-derated LUT through the wrong (unrelated, pre-noActivation-choice)
    avg_bits table, same failure mode _HLS_MVU_LUT_MULT_DERATE's own
    lut_mult bypass exists to avoid."""
    return variant_kwargs["force_dsp"] or variant_kwargs["impl_style"] == IMPL_STYLE_RTL


def load_config(config_module: str) -> None:
    """Injects the named config_*.py's constants into this module's
    globals (IN_CHANNELS, CHANNELS, BOTTLENECKS_PER_STAGE, ... -- everything
    ENet(...)'s call in main() below reads). Always called from main();
    there is no built-in default config any more (see module docstring's
    2026-09-17 REFACTOR note -- the old config_23_1.py default is archived)."""
    cfg = importlib.import_module(config_module)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def _pair(v) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)


def trace_layer_geometry(model: torch.nn.Module, input_hw: tuple[int, int], in_channels: int) -> tuple[list[LayerGeometry], list[str]]:
    """Forward-hook trace of every real Conv2d/ConvTranspose2d/MaxPool2d in
    `model`, tagged with its owning bottleneck BLOCK name (via
    block_utils.path_to_block_map's exact full-path lookup) -- purely for
    traceability/reporting on each per_layer output entry's own "stage"
    field, not load-bearing for bit-width or folding linking (both are
    indexed by individual layer name, not block, throughout this file)."""
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
                groups=getattr(module, "groups", 1),
            ))
        return hook

    handles = []
    for name, module in model.named_modules():
        block_name = path_to_block.get(name)
        if block_name is None:
            continue
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "Conv2d")))
        elif isinstance(module, torch.nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "ConvTranspose2d")))
        elif isinstance(module, torch.nn.MaxPool2d):
            handles.append(module.register_forward_hook(make_hook(name, block_name, "MaxPool2d")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, *input_hw))
    for h in handles:
        h.remove()
    return geometries, list(blocks.keys())


def candidate_folds(layer: LayerGeometry) -> list[tuple[int, int, str, str]]:
    """Every valid (PE, SIMD, ram_style, variant) 4-tuple for this layer --
    MaxPool2d has neither PE/SIMD/ram_style/variant (no MVAU, no weights,
    no activation to fuse), so it gets the single sentinel
    (1, 1, "block", VARIANT_RTL_DSP_NOACT1), which layer_cost_pe_simd
    ignores for that op_type anyway (its cost/cycles don't depend on any of
    those at all -- see maxpool_cost).

    ram_style (2026-09-18): drives block-vs-distributed for the standalone
    Thresholding_rtl node's OWN memory only (thr_ram_style) -- a real, free
    per-layer LUT-vs-BRAM tradeoff. MVAU's own weight memory and SWU's own
    line-buffer memory both have NO free choice left at all (hard-fixed to
    block and distributed respectively) -- see RAM_STYLES' own module-level
    comment for the full history and evidence on all three consumers, URAM's
    now-permanent retirement everywhere, and why "distributed" is real for
    Thresholding specifically (structurally synthesis-safe, just never yet
    exercised in real hardware) unlike "ultra" (a confirmed real synthesis
    failure). See finn_cost_model.py's _thresholding_rtl_cost and
    _THR_RTL_LUTRAM_PER_PE_NUMSTEP docstrings for the cost formula itself.
    FIFOs remain a separate, NOT-YET-modeled resource in this cost model
    (see finn_cost_model.py's own top-of-file "Still not covered" list) --
    their own real ram_style choice is not part of this ILP at all yet.

    FORCE_SERIAL (set via --force-serial) restricts every layer to (1, 1).

    variant (2026-09-17 addition, additive/opt-in): VARIANT_RTL_DSP_NOACT1 is
    always eligible for every Conv2d/ConvTranspose2d layer (today's only
    variant, unchanged) -- INCLUDING depthwise ones: conv_cost_pe_simd
    itself already silently overrides impl_style to HLS for depthwise
    layers regardless of what's requested (finn_cost_model.py:686-687,
    VVAU_rtl needs a Versal DSP58), so requesting VARIANT_RTL_DSP_NOACT1
    for a depthwise layer already resolves to exactly today's real
    behavior (HLS + force_dsp + noActivation=1) -- excluding it here would
    silently change depthwise layers' candidate set instead of preserving
    it. VARIANT_HLS_LUT_NOACT0 is eligible for every layer too, but only
    ADDED to the candidate set when ALLOW_LUT_MULT is set
    (--allow-lut-mult) -- with it unset (the default), every layer's
    candidate set is IDENTICAL to before this addition, just each tuple now
    carries an extra, constant VARIANT_RTL_DSP_NOACT1 tag. Note: RTL's
    further bit-width eligibility (weights signed, 2<=w<=8, a<=8) can't be
    checked here -- candidate_folds doesn't see (w,a) at all -- it's
    filtered in solve_joint_perlayer's own (w,a) inner loop instead."""
    if layer.op_type == "MaxPool2d":
        return [(1, 1, "block", VARIANT_RTL_DSP_NOACT1)]
    variants = [VARIANT_RTL_DSP_NOACT1]
    if ALLOW_LUT_MULT:
        variants.append(VARIANT_HLS_LUT_NOACT0)
    folds = [
        (pe, simd, ram_style, variant)
        for pe in divisors(max_pe(layer)) for simd in divisors(max_simd(layer))
        for ram_style in RAM_STYLES for variant in variants
    ]
    if FORCE_SERIAL:
        folds = [f for f in folds if f[0] == 1 and f[1] == 1]
    return folds


def _normalize(values: dict[tuple, float]) -> dict[tuple, float]:
    """Plain min-max scale to [0,1] -- puts sensitivity and cycles, which
    differ by several orders of magnitude, on a comparable footing so
    alpha is a meaningful single knob rather than requiring the caller to
    know each quantity's raw scale. (A trimmed copy of the old ilp_search.py
    _normalize: that version also supported a robust-percentile clip and a
    log-space option for the per-BLOCK/per-STAGE search's own outlier
    problems -- neither is exercised by this file, which only ever calls
    _normalize with its defaults, so only the plain min-max path is kept.)"""
    vals = values.values()
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


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
    force_dsp: bool = False,
    require_simd_ge_pe: bool = False,
    hard_dsp_fraction: float = 1.0,
    hard_uram_fraction: float = 1.0,
    max_join_imbalance_ratio: float | None = None,
) -> dict:
    """The per-layer combined MILP -- see module docstring for the full
    formulation.

    predecessor_map: from layer_topology.compute_predecessor_map (or None to
    skip the correction entirely, falling back to self-indexed act
    sensitivity for every layer) -- see module docstring's PREDECESSOR-
    CORRECTED ACT SENSITIVITY section for what this fixes and
    _act_sensitivity_sources for the exact fallback rules.

    max_join_imbalance_ratio: hard per-join balance constraint (None =
    disabled, prior behavior exactly). A "join" is any layer with 2+ real
    predecessors in `predecessor_map` -- two (or more) PARALLEL branches
    that execute CONCURRENTLY in real FINN dataflow hardware, reconverging
    at a residual add/concat. Nothing else in this ILP is aware layers can
    be siblings this way -- each layer's own (PE,SIMD,bits) is chosen with
    zero visibility into its join partner, and a large skew is exactly what
    forces a real, UNMODELED FIFO in actual hardware (see finn_cost_model.py's
    own "still not covered: FIFOs" note and MILP/scan_fork_join_mismatch.py's
    `predicted_depth`). For every real join (deduplicated by its own
    predecessor set), every ordered pair of branches (i, j) gets
    `cycles[i] <= max_join_imbalance_ratio * cycles[j]`. MaxPool2d branches
    are EXEMPT (candidate_folds gives them a single fixed (1,1) fold -- no
    real folding choice to trade off, so balancing a conv branch against one
    always forces over-folding regardless of ratio; only pairs where BOTH
    branches have a real folding decision get balanced). Does NOT add or
    price any FIFO resource itself -- it only prevents the solver from
    choosing a badly skewed pair in the first place."""
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

    # -- z[layer,pe,simd,ram_style,variant,w,a]: cycles/LUT/BRAM/DSP attach
    # here, at the EXACT (fold,variant,bits) combination. `variant` is the
    # 2026-09-17 addition (see VARIANT_RTL_DSP_NOACT1/VARIANT_HLS_LUT_NOACT0
    # above) -- additive/opt-in, see candidate_folds' own docstring: with
    # ALLOW_LUT_MULT unset (the default) every layer's folds list carries
    # only VARIANT_RTL_DSP_NOACT1, so this loop and everything downstream is
    # BYTE-IDENTICAL to before this addition (force_dsp resolves to the same
    # `force_dsp` argument via _variant_cost_kwargs).
    z: dict[tuple, pulp.LpVariable] = {}
    layer_costs: dict[tuple, dict] = {}
    raw_cycles: dict[tuple, float] = {}
    raw_lut: dict[tuple, float] = {}
    raw_bram: dict[tuple, float] = {}
    raw_dsp: dict[tuple, float] = {}
    raw_uram: dict[tuple, float] = {}
    layer_folds: dict[str, list[tuple[int, int, str, str]]] = {}
    # Per-layer RAW (un-normalized) cycle expression -- sum over that
    # layer's own chosen (fold,variant,bits) z-variable weighted by its own
    # raw_cycles -- built alongside z/raw_cycles below and reused by the
    # join-balance constraints (max_join_imbalance_ratio) since it needs
    # each layer's cycles as one linear pulp expression, the exact same
    # quantity latency_term/max_cycles already sum across every layer.
    layer_cycle_terms: dict[str, list[tuple[pulp.LpVariable, float]]] = {g.name: [] for g in geometries}

    for layer in geometries:
        folds = candidate_folds(layer)  # respects module-level FORCE_SERIAL/ALLOW_LUT_MULT if set
        if require_simd_ge_pe:
            # Hard structural fix for the PE>>SIMD real-LUT blowup documented in
            # hardware/mvau_lut_correlation_report.txt: 36% of the 89-row real
            # calibration dataset has PE>SIMD, yet those rows account for 70.7%
            # of total real_LUT. Rules out that whole region up front instead of
            # relying on a fitted imbalance_luts penalty to price it correctly
            # (that term has since been REMOVED from finn_cost_model.py -- see
            # its own module docstring's 2026-09-17 refresh). PE=1 is always
            # paired with SIMD=max_simd(layer)>=1 by candidate_folds, so this
            # never empties a layer's fold set. (f[1]=simd, f[0]=pe -- unaffected
            # by the variant element now at f[3].)
            folds = [f for f in folds if f[1] >= f[0]]
        layer_folds[layer.name] = folds
        for pe, simd, ram_style, variant in folds:
            variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
            for w, a in candidate_pairs:
                # MVAU's own weight memory is hard-fixed to block (real hardware never
                # materialized "ultra"). SWU's own line-buffer memory is hard-fixed to
                # distributed (48/48 real nodes, regardless of what was requested). Neither
                # has a real free choice left -- see RAM_STYLES' own comment above for the
                # full story/evidence on both. The `ram_style` loop variable now drives
                # block-vs-distributed for the standalone Thresholding_rtl node's own
                # memory (thr_ram_style) -- a real, free per-layer LUT-vs-BRAM tradeoff
                # (see RAM_STYLES' own comment for why "distributed" is real here but
                # "ultra" never was).
                cost = layer_cost_pe_simd(
                    layer, w, a, pe, simd, RAM_STYLE_BLOCK,
                    swu_ram_style="distributed", thr_ram_style=ram_style, **variant_kwargs,
                )
                key = (layer.name, pe, simd, ram_style, variant, w, a)
                layer_costs[key] = cost
                raw_cycles[key] = cost["cycles"]
                raw_lut[key] = calibrated_lut(
                    cost["total_lut"], w, a, force_dsp=_calibration_force_dsp(variant_kwargs),
                    lut_mult=(variant == VARIANT_HLS_LUT_NOACT0),
                )
                # thr_bram18: the standalone Thresholding_rtl's own memory (noActivation=1 regime, see
                # finn_cost_model.conv_cost_pe_simd) -- absent for MaxPool2d, and 0 for VARIANT_HLS_LUT_NOACT0
                # (no separate node -- see that function's no_activation=False docstring).
                raw_bram[key] = calibrated_bram18k(
                    cost["swu_bram18"] + cost["wm_bram18"] + cost.get("thr_bram18", 0), w, a,
                    force_dsp=variant_kwargs["force_dsp"],
                )
                raw_dsp[key] = cost["total_dsp"]
                # wm_uram18 is always 0 (MVAU ram_style hard-fixed to block).
                # swu_uram18 is always 0 too (SWU ram_style hard-fixed to distributed,
                # 2026-09-18 -- see the cost call's own comment above for why).
                # thr_uram18 comes from the `ram_style` loop variable (Thresholding's
                # own real, free ILP choice) -- see candidate_folds' docstring.
                raw_uram[key] = cost.get("wm_uram18", 0) + cost.get("swu_uram18", 0) + cost.get("thr_uram18", 0)
                z[key] = pulp.LpVariable(f"z_{layer.name}_{pe}_{simd}_{ram_style}_{variant}_{w}_{a}", cat=pulp.LpBinary)
                layer_cycle_terms[layer.name].append((z[key], raw_cycles[key]))

    cycles_norm = _normalize(raw_cycles)
    layer_cycles_expr: dict[str, pulp.LpAffineExpression] = {
        name: pulp.lpSum(zvar * cyc for zvar, cyc in terms) for name, terms in layer_cycle_terms.items()
    }

    prob = pulp.LpProblem("FINN_MILP_perlayer", pulp.LpMinimize)

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

    # Linking constraint -- a same-layer identity.
    for layer in geometries:
        folds = layer_folds[layer.name]
        for w, a in candidate_pairs:
            prob += (
                pulp.lpSum(z[(layer.name, pe, simd, ram_style, variant, w, a)] for pe, simd, ram_style, variant in folds)
                == y[(layer.name, w, a)]
            ), f"link_{layer.name}_{w}_{a}"

    # Join-balance constraints (max_join_imbalance_ratio). Dedupe joins by
    # their own predecessor SET first: predecessor_map is keyed by CONSUMER
    # layer, so multiple consumers of the same join (e.g. both of down1's
    # own two downstream layers) independently map to the SAME branch pair
    # -- that is one real join, not two. MaxPool2d branches are EXEMPT (see
    # this function's own docstring) -- only pairs where BOTH sides have a
    # real folding decision get balanced.
    maxpool_names = {g.name for g in geometries if g.op_type == "MaxPool2d"}
    n_join_constraints = 0
    if max_join_imbalance_ratio is not None:
        if predecessor_map is None:
            raise ValueError(
                "max_join_imbalance_ratio requires a predecessor_map (pass predecessor_map=... -- "
                "see layer_topology.compute_predecessor_map) -- without it there is no join information "
                "to balance."
            )
        unique_joins = {
            tuple(sorted(preds)) for preds in predecessor_map.values()
            if len(preds) >= 2 and all(p in layer_cycles_expr for p in preds)
        }
        for branches in unique_joins:
            foldable_branches = [b for b in branches if b not in maxpool_names]
            for branch_i in foldable_branches:
                for branch_j in foldable_branches:
                    if branch_i == branch_j:
                        continue
                    prob += (
                        layer_cycles_expr[branch_i] <= max_join_imbalance_ratio * layer_cycles_expr[branch_j]
                    ), f"join_balance_{branch_i}_vs_{branch_j}"
                    n_join_constraints += 1

    # Hard resource constraints.
    prob += pulp.lpSum(z[k] * raw_lut[k] for k in z) <= hard_lut_fraction * XCZU7EV["LUT"], "hard_lut_budget"
    prob += pulp.lpSum(z[k] * raw_bram[k] for k in z) <= hard_bram_fraction * XCZU7EV["BRAM_18K"], "hard_bram_budget"
    # DSP: real hard constraint too (2026-09-17). Uncapped, the RTL packing
    # rule (ceil(PE/lanes)*SIMD) let the S12 dense alpha=0.25 solve pick a
    # plan needing 2564 DSP48s on a 1728-DSP device.
    prob += pulp.lpSum(z[k] * raw_dsp[k] for k in z) <= hard_dsp_fraction * XCZU7EV["DSP"], "hard_dsp_budget"
    # URAM: real hard constraint too (2026-09-17). Previously UNCAPPED --
    # the S12 dense alpha=0.25 solve picked wm_uram18=205 total (an
    # unconstrained "ultra" ram_style choice with zero resource pressure
    # pushing back), while the real 8-way hardware build for that exact
    # solve landed at URAM=2/96 -- either the ILP's own ram_style choice was
    # never applied to the real per-node build (a bridge gap, see thr_pe's
    # own analogous note), or an unconstrained URAM axis alone can already
    # produce plans far past the real 96-block budget. Capping it here fixes
    # the latter regardless of the former. XCZU7EV["URAM"]=96 real URAM288
    # blocks (96*288Kib=27Mib) on xczu7ev-ffvc1156-2-e.
    prob += pulp.lpSum(z[k] * raw_uram[k] for k in z) <= hard_uram_fraction * XCZU7EV["URAM"], "hard_uram_budget"
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
        raise RuntimeError(f"per-layer ILP hit an unexpected solver status: {status_name!r}.")

    n_binary_vars = len(y) + len(z)
    n_constraints = (
        n_layers + sum(len(candidate_pairs) for _ in geometries) + 4
        + (1 if max_cycles is not None else 0) + n_join_constraints
    )

    if status_name != "Optimal":
        return {
            "status": status_name,
            "alpha": alpha,
            "layer_weight_bits": {}, "layer_act_bits": {}, "per_layer": {},
            "_diagnostics": {
                "alpha": alpha, "candidate_bits": list(CANDIDATE_BITS), "n_layers": n_layers,
                "n_binary_vars": n_binary_vars, "n_constraints": n_constraints,
                "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
                "hard_dsp_fraction": hard_dsp_fraction, "hard_uram_fraction": hard_uram_fraction,
                "max_cycles": max_cycles,
                "max_join_imbalance_ratio": max_join_imbalance_ratio, "n_join_constraints": n_join_constraints,
                "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": FORCE_SERIAL,
                "force_dsp": force_dsp, "require_simd_ge_pe": require_simd_ge_pe, "allow_lut_mult": ALLOW_LUT_MULT,
                "note": f"Solver status {status_name!r} -- no joint per-layer (bits, folding) assignment "
                        f"satisfies the requested hard LUT/BRAM/DSP/URAM budget(s) (hard_lut_fraction={hard_lut_fraction}, "
                        f"hard_bram_fraction={hard_bram_fraction}, hard_dsp_fraction={hard_dsp_fraction}, "
                        f"hard_uram_fraction={hard_uram_fraction})"
                        + (f" and max_cycles={max_cycles:.0f}" if max_cycles is not None else "")
                        + (f" and max_join_imbalance_ratio={max_join_imbalance_ratio}" if max_join_imbalance_ratio is not None else "")
                        + " at all.",
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
        for pe, simd, ram_style, variant in folds:
            if pulp.value(z[(layer.name, pe, simd, ram_style, variant, w, a)]) > 0.5:
                chosen_fold = (pe, simd, ram_style, variant)
                break
        assert chosen_fold is not None, f"layer {layer.name}: no folding choice selected at its own chosen (w,a)={w,a}"
        pe, simd, ram_style, variant = chosen_fold
        cost = layer_costs[(layer.name, pe, simd, ram_style, variant, w, a)]
        variant_kwargs = _variant_cost_kwargs(variant, force_dsp)
        per_layer[layer.name] = {
            # thr_ram_style (renamed from "ram_style" 2026-09-18): this loop
            # variable/z-key element only ever drives the standalone
            # Thresholding_rtl node's own memory now (see candidate_folds'
            # docstring) -- MVAU's own weight memory is hard-fixed to block,
            # SWU's own line-buffer memory is hard-fixed to distributed.
            # "ram_style" was a stale, misleadingly-generic export name.
            "stage": layer.stage, "pe": pe, "simd": simd, "thr_ram_style": ram_style, "variant": variant,
            # force_dsp/mvau_noAct: the same two axes packed into `variant`
            # (impl_style is already its own field, from **cost below), split
            # out explicitly for readability -- `variant` itself is KEPT (not
            # replaced) since it's still used internally below (the per-layer
            # force_dsp lookup for calibrated_lut/calibrated_bram18k re-derives
            # this same variant_kwargs from `variant` again, see the comment
            # a few lines down) and as the z-key/candidate_folds identity.
            "force_dsp": variant_kwargs["force_dsp"], "mvau_noAct": variant_kwargs["no_activation"],
            "weight_bits": w, "act_bits": a, **cost,
        }

    # Per-layer force_dsp for the post-hoc calibration below MUST follow each
    # layer's OWN chosen variant, not the single global `force_dsp` CLI flag
    # -- with ALLOW_LUT_MULT unset every v["variant"] is VARIANT_RTL_DSP_NOACT1,
    # for which _variant_cost_kwargs(...)["force_dsp"] == force_dsp exactly,
    # so this is byte-identical to before for every existing invocation.
    total_lut = sum(
        calibrated_lut(
            v["total_lut"], v["weight_bits"], v["act_bits"],
            force_dsp=_calibration_force_dsp(_variant_cost_kwargs(v["variant"], force_dsp)),
            lut_mult=(v["variant"] == VARIANT_HLS_LUT_NOACT0),
        )
        for v in per_layer.values()
    )
    total_bram = sum(
        calibrated_bram18k(
            v["swu_bram18"] + v["wm_bram18"] + v.get("thr_bram18", 0), v["weight_bits"], v["act_bits"],
            force_dsp=_variant_cost_kwargs(v["variant"], force_dsp)["force_dsp"],
        )
        for v in per_layer.values()
    )
    total_uram = sum(
        v.get("wm_uram18", 0) + v.get("swu_uram18", 0) + v.get("thr_uram18", 0) for v in per_layer.values()
    )
    total_dsp = sum(v["total_dsp"] for v in per_layer.values())
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
            "total_uram18": total_uram, "xczu7ev_uram_budget": XCZU7EV["URAM"],
            "uram_pct_of_budget": 100 * total_uram / XCZU7EV["URAM"], "total_cycles": total_cycles,
            "total_dsp": total_dsp, "xczu7ev_dsp_budget": XCZU7EV["DSP"],
            "dsp_pct_of_budget": 100 * total_dsp / XCZU7EV["DSP"],
            "hard_lut_fraction": hard_lut_fraction, "hard_bram_fraction": hard_bram_fraction,
            "hard_dsp_fraction": hard_dsp_fraction, "hard_uram_fraction": hard_uram_fraction,
            "max_cycles": max_cycles,
            "max_join_imbalance_ratio": max_join_imbalance_ratio, "n_join_constraints": n_join_constraints,
            "solver_time_limit_s": time_limit, "solver_gap_rel": gap_rel, "force_serial": FORCE_SERIAL,
            "force_dsp": force_dsp, "require_simd_ge_pe": require_simd_ge_pe, "allow_lut_mult": ALLOW_LUT_MULT,
            "note": "Joint per-LAYER MILP: y[layer,w,a] (sensitivity) linked to z[layer,pe,simd,ram_style,w,a] "
                    "(cycles/LUT/BRAM/DSP/URAM) via a same-layer equality constraint -- LUT/BRAM/DSP/URAM are REAL "
                    "hard <= XCZU7EV constraints here (not a soft penalty). ram_style selects the SWU's own line-"
                    "buffer memory (2026-09-17: MVAU's own weight-memory ram_style is hard-fixed to block, ultra "
                    "disabled -- real hardware never used it, see candidate_folds' own docstring). Objective = "
                    "alpha*mean(sens_norm) + (1-alpha)*mean(cycles_norm), both mean-normalized by the SAME "
                    "n_layers. GUARANTEED to fit the requested hard budget(s) under this cost model's own "
                    "calibration -- a steering signal at that calibration, not a certified hardware guarantee. "
                    "Coarser than nnunetv2.nets.LayerQuantENet's own full per-quantizer-site deployment schema -- "
                    "see module docstring's SCOPE BOUNDARY.",
        },
    }


_SUMMARY_FIELDS = [
    "alpha", "status", "avg_weight_bits", "avg_act_bits",
    "lut_pct_of_budget", "bram_pct_of_budget", "dsp_pct_of_budget",
    "total_dsp", "total_cycles", "clock_mhz", "latency_ms",
    "n_binary_vars", "n_layers",
]


def _update_sweep_summary(out_dir: Path, args: argparse.Namespace, result: dict) -> None:
    """Maintains summary.csv and run_args.json in out_dir across an ALPHA
    SWEEP -- a 5-alpha sweep normally run as 5 separate `finn_milp.py`
    invocations (see e.g. compression/slurm/sensitivity_ilp_*.job's own
    per-alpha loop), each one upserting its own row/shared-args here, so
    the sweep ends up with one coherent, always-current summary with no
    separate aggregation script or manual step needed.

    summary.csv: one row per alpha, replaced (not duplicated) on rerun --
    keyed by the `alpha` column, matching upsert_row's own config_name-keyed
    replace semantics in compression/collect_results.py. An existing row
    from an OLDER schema version (e.g. missing dsp_pct_of_budget, before
    this repo's forced-DSP-regime hard DSP cap existed) is dropped rather
    than partially merged -- summary.csv is a derived artifact, safe to
    regenerate wrong-then-right one alpha at a time as each is rerun.

    run_args.json: shared_args should be IDENTICAL across every alpha of one
    real sweep -- if an existing run_args.json's shared_args disagree with
    this invocation's, that's flagged loudly (a real inconsistency, e.g.
    someone changed --hard-lut-fraction between alphas) rather than
    silently overwritten without comment."""
    diag = result["_diagnostics"]
    weight_bits, act_bits = result.get("layer_weight_bits", {}), result.get("layer_act_bits", {})
    avg_weight_bits = sum(weight_bits.values()) / len(weight_bits) if weight_bits else float("nan")
    avg_act_bits = sum(act_bits.values()) / len(act_bits) if act_bits else float("nan")
    total_cycles = diag.get("total_cycles")
    latency_ms = total_cycles / (args.clock_mhz * 1000) if total_cycles is not None else float("nan")

    row = {
        "alpha": args.alpha, "status": result["status"],
        "avg_weight_bits": avg_weight_bits, "avg_act_bits": avg_act_bits,
        "lut_pct_of_budget": diag.get("lut_pct_of_budget"), "bram_pct_of_budget": diag.get("bram_pct_of_budget"),
        "dsp_pct_of_budget": diag.get("dsp_pct_of_budget"), "total_dsp": diag.get("total_dsp"),
        "total_cycles": total_cycles, "clock_mhz": args.clock_mhz, "latency_ms": latency_ms,
        "n_binary_vars": diag.get("n_binary_vars"), "n_layers": diag.get("n_layers"),
    }

    summary_path = out_dir / "summary.csv"
    rows = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if float(r["alpha"]) != args.alpha]
    rows.append(row)
    rows.sort(key=lambda r: float(r["alpha"]))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {summary_path} ({len(rows)} alpha rows).")

    run_args_path = out_dir / "run_args.json"
    shared_args = {
        "config": args.config, "sensitivity-file": str(args.sensitivity_file),
        "candidate-bits": ",".join(str(b) for b in CANDIDATE_BITS),
        "hard-lut-fraction": args.hard_lut_fraction, "hard-bram-fraction": args.hard_bram_fraction,
        "hard-dsp-fraction": args.hard_dsp_fraction, "hard-uram-fraction": args.hard_uram_fraction, "force-dsp": args.force_dsp,
        "max-latency-ms": args.max_latency_ms, "clock-mhz": args.clock_mhz,
        "max-join-imbalance-ratio": args.max_join_imbalance_ratio,
        "force-serial": FORCE_SERIAL, "require-simd-ge-pe": args.require_simd_ge_pe,
        "allow-lut-mult": ALLOW_LUT_MULT,
        "time-limit": args.time_limit, "gap-rel": args.gap_rel,
    }
    existing_alphas = []
    if run_args_path.exists():
        existing = json.loads(run_args_path.read_text())
        existing_alphas = existing.get("alphas", [])
        prior_shared = existing.get("shared_args", {})
        mismatched = {k: (prior_shared[k], v) for k, v in shared_args.items() if k in prior_shared and prior_shared[k] != v}
        if mismatched:
            print(f"WARNING: {run_args_path} was last written with DIFFERENT shared args for other alpha(s) in "
                  f"this sweep: {mismatched} -- overwriting shared_args with THIS run's values; the sweep may "
                  f"now be inconsistent across alphas (mixed hard caps/config/etc).")
    run_args = {
        "pipeline": [
            "MILP/layer_sensitivity.py --candidate-bits ...",
            "MILP/finn_milp.py --candidate-bits ... --hard-lut-fraction ... --hard-bram-fraction ... "
            "--hard-dsp-fraction ... --force-dsp",
            "MILP/expand_layer_bits.py",
        ],
        "shared_args": shared_args,
        "alphas": sorted(set(existing_alphas) | {args.alpha}),
        "granularity": "layer",
        "notes": f"Per-layer joint bits+folding MILP for {args.config}, hard LUT/BRAM/DSP caps "
                 f"{args.hard_lut_fraction}/{args.hard_bram_fraction}/{args.hard_dsp_fraction} under "
                 f"force_dsp={args.force_dsp}. See summary.csv for per-alpha results.",
    }
    run_args_path.write_text(json.dumps(run_args, indent=2))
    print(f"Updated {run_args_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                         help="Which MILP/config_*.py to load -- e.g. config_12_dense_relu_warmstart150ep.")
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
    parser.add_argument("--hard-dsp-fraction", type=float, default=1.0,
                         help="Same as --hard-lut-fraction, for DSP48E2 slices (XCZU7EV['DSP']=1728). Always enforced.")
    parser.add_argument("--hard-uram-fraction", type=float, default=1.0,
                         help="Same as --hard-lut-fraction, for URAM288 blocks (XCZU7EV['URAM']=96, 27Mib). "
                              "Currently INERT: total_uram18 is structurally 0 for every layer, always, since "
                              "2026-09-18 -- MVAU weight memory, SWU line buffer, and Thresholding accumulator "
                              "ROM (the three consumers this axis has driven, one at a time) each turned out to "
                              "have no real, working URAM path on this device (xczu7ev, UltraScale+): real "
                              "hardware never materialized MVAU's own 'ultra' request, 48/48 real SWU nodes used "
                              "'distributed' regardless of what was requested, and a real Vivado synthesis "
                              "attempt of Thresholding_rtl with ram_style='ultra' FAILS outright (URAM288 cannot "
                              "be used as ROM). See RAM_STYLES' own comment for the full history. Kept, not "
                              "removed, for provenance and in case a genuinely URAM-eligible consumer is added.")
    parser.add_argument("--force-dsp", action="store_true",
                         help="Cost every (layer, fold, bits) combination under the FORCED-DSP regime's own "
                              "flat empirical LUT/BRAM factor (finn_cost_model.py's _FORCED_DSP_LUT_FACTOR/"
                              "_FORCED_DSP_BRAM_FACTOR, currently identity -- see that file's own module "
                              "docstring) instead of the default auto-resType avg_bits-interpolated table.")
    parser.add_argument("--max-latency-ms", type=float, default=None,
                         help="Hard cap on total cycles, expressed as a latency budget at --clock-mhz (default "
                              "None = no cap). Converted to max_cycles = max_latency_ms/1000 * clock_mhz*1e6 and "
                              "added as a real <= constraint.")
    parser.add_argument("--clock-mhz", type=float, default=100.0,
                         help="Clock frequency --max-latency-ms is expressed against (default 100.0).")
    parser.add_argument("--max-join-imbalance-ratio", type=float, default=None,
                         help="Hard per-join balance constraint (default None = no constraint, prior behavior "
                              "exactly). A 'join' is any point with 2+ real predecessors (layer_topology."
                              "compute_predecessor_map) -- i.e. branches that run CONCURRENTLY in real FINN "
                              "dataflow hardware. Without this, each branch's own (PE,SIMD,bits) is chosen with "
                              "zero awareness of its join partner, so the solver can leave a large cycle-count "
                              "skew behind -- exactly what forces a real, UNMODELED FIFO in actual hardware "
                              "(the faster branch has to buffer while waiting on the slower one; see "
                              "MILP/scan_fork_join_mismatch.py's own `predicted_depth`, which estimates this "
                              "post-hoc). Setting this to e.g. 1.5 requires every join's own branches to stay "
                              "within a 1.5x cycle-count ratio of each other. MaxPool2d branches are EXEMPT "
                              "(fixed (1,1) fold, no real folding choice to trade off -- only conv-vs-conv join "
                              "pairs get balanced). Does not price the FIFO itself -- only prevents picking a "
                              "badly skewed pair.")
    parser.add_argument("--force-serial", action="store_true",
                         help="Force FOLDING_SERIAL (PE=SIMD=1) on every layer before solving.")
    parser.add_argument("--allow-lut-mult", action="store_true",
                         help="Additive/opt-in (2026-09-17): let the ILP choose, PER LAYER, a new "
                              "'hls_lut_noact0' resource variant (HLS backend, LUT-based multiplication instead "
                              "of DSP, fused activation instead of a separate Thresholding node) alongside "
                              "today's default 'rtl_dsp_noact1' variant, to spend spare LUT budget instead of "
                              "DSP budget on specific layers. Default False preserves prior behavior EXACTLY "
                              "(every layer stays 'rtl_dsp_noact1' only). The new variant's fused-threshold LUT "
                              "term is a direct FINN-source transcription, not independently calibrated against "
                              "real hardware yet -- see finn_cost_model.py's conv_cost_pe_simd docstring.")
    parser.add_argument("--require-simd-ge-pe", action="store_true",
                         help="Drop every (PE,SIMD) candidate fold with PE>SIMD before solving -- a hard, "
                              "zero-fitted-parameter fix for the real-LUT blowup documented in "
                              "hardware/mvau_lut_correlation_report.txt (PE>SIMD rows are 36%% of the real "
                              "calibration dataset but 70.7%% of its total real_LUT). Default False preserves "
                              "prior behavior exactly.")
    parser.add_argument("--time-limit", type=int, default=1800,
                         help="CBC wall-clock cap in seconds (default 1800 = 30min).")
    parser.add_argument("--gap-rel", type=float, default=0.02,
                         help="CBC relative optimality gap to accept (default 0.02 = accept anything CBC can "
                              "prove is within 2%% of the true optimum).")
    parser.add_argument("--pin-bits-file", type=Path, default=None,
                         help="TEST-ONLY: a layer_bits_*.json ({'layer_weight_bits': {...}, 'layer_act_bits': "
                              "{...}}) to pin y to instead of letting alpha decide. --alpha is still required "
                              "but has no effect on the result when set.")
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    load_config(args.config)

    if args.candidate_bits is not None:
        global CANDIDATE_BITS
        CANDIDATE_BITS = tuple(sorted(int(b) for b in args.candidate_bits.split(",")))
        print(f"Overriding CANDIDATE_BITS to {CANDIDATE_BITS} (from --candidate-bits).")

    if args.force_serial:
        global FORCE_SERIAL
        FORCE_SERIAL = True
        print("--force-serial: every layer restricted to (PE, SIMD) = (1, 1) before solving.")

    if args.allow_lut_mult:
        global ALLOW_LUT_MULT
        ALLOW_LUT_MULT = True
        print("--allow-lut-mult: 'hls_lut_noact0' (LUT-mult, fused activation) made eligible alongside "
              "'rtl_dsp_noact1' on every layer -- PROVISIONAL fused-threshold LUT term, see --help.")

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
    geometries, _block_names = trace_layer_geometry(model, INPUT_HW, IN_CHANNELS)
    layer_names = tuple(g.name for g in geometries)

    # Predecessor-corrected act sensitivity -- see module docstring's
    # PREDECESSOR-CORRECTED ACT SENSITIVITY section. Never fatal: any
    # tracing failure falls back to self-indexed act sensitivity for EVERY
    # layer, rather than crashing the whole run.
    try:
        predecessor_map = compute_predecessor_map(model)
        n_resolved = sum(1 for name in layer_names if predecessor_map.get(name))
        print(f"Predecessor map: {n_resolved}/{len(layer_names)} layers have a real, traced predecessor "
              f"(the rest are the network's own first layer(s), with no real predecessor to fall back to "
              f"anything but self-sensitivity).")
    except Exception as error:
        predecessor_map = None
        print(f"WARNING: could not compute a predecessor map ({type(error).__name__}: {error}) -- falling back "
              f"to self-indexed act sensitivity for EVERY layer.")

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
    n_folds_per_layer = [
        len([f for f in candidate_folds(g) if f[1] >= f[0]]) if args.require_simd_ge_pe
        else len(candidate_folds(g))
        for g in geometries
    ]
    n_z = sum(n_folds_per_layer) * candidate_pairs_count
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
        predecessor_map=predecessor_map, force_dsp=args.force_dsp, require_simd_ge_pe=args.require_simd_ge_pe,
        hard_dsp_fraction=args.hard_dsp_fraction, hard_uram_fraction=args.hard_uram_fraction,
        max_join_imbalance_ratio=args.max_join_imbalance_ratio,
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
    print(f"DSP used: {diag['total_dsp']:.0f} ({diag['dsp_pct_of_budget']:.1f}% of {XCZU7EV['DSP']} budget) "
          f"-- GUARANTEED (hard constraint).")
    print(f"Total cycles (sum, ~= per-image latency): {diag['total_cycles']:.0f}")

    _update_sweep_summary(args.out_file.parent, args, result)


if __name__ == "__main__":
    main()
