"""Analytical FINN dataflow resource-cost formulae, parameterized by weight
bit-width W, activation bit-width A, AND folding config (P, Q -- how many
output channels / reduction elements are computed in parallel per cycle),
instead of hardcoded 8x8 bits and a single fixed folding.

These are the SAME formulae already used to produce this repo's real FINN
estimate reports (e.g. hardware/outputs/quantEnet_original_int8_unfolded_
report/files/enet_finn_fully_unfolded_M1_stage_summary.csv and its sibling
finn_cost_formulae.md, source: Blott et al., "FINN-R: An End-to-End
Deep-Learning Framework for Fast Exploration of Quantized Neural Networks",
ACM TRETS 2018, Sec. 3.2) -- verified by hand against that report's own
per-layer CSV (e.g. down1.reduce.0: cin=16,cout=16,kh=kw=sh=sw=2,W=A=8 ->
wm_bram18=240, swu_bram18=8, mvu_lut=72390, matches this module's formulas
exactly, at FOLDING_UNFOLDED). Reimplemented here as pure functions of
(W, A, folding) rather than reading that fixed-8x8/fixed-folding report,
since the HAWQ per-stage search needs the SAME cost model evaluated at
W,A in {2,4,8} -- no FINN toolchain/Docker container needed for this, it's
a closed-form estimate, not an actual FINN build.

Two folding configs, the two ends of the P/Q spectrum:
  FOLDING_UNFOLDED (Q=C_in*K_h*K_w, P=C_out): the "fully unfolded" case
    every existing report/estimate in this repo uses -- entire reduction
    and all output channels computed in ONE cycle per output pixel.
    Maximum resource usage, minimum latency.
  FOLDING_SERIAL (Q=1, P=1): the most serial case -- one reduction element
    and one output channel per cycle. Minimum resource usage, maximum
    latency (this is what compression/hawq/ analysis was asked to check:
    does going maximally serial let the design fit real BRAM/LUT budgets).
Both use the SAME general BRAM_wm formula (omega = K^2*C*C'/(Q*P), not the
"omega=1" shortcut a folding-unaware version would need) -- confirmed this
still reduces to the exact verified FOLDING_UNFOLDED numbers above (Q*P
always equals the full weight volume there, so omega=1 falls out
automatically, byte-identical to the old hardcoded-omega=1 formula).

IMPORTANT finding from comparing the two: BRAM_swu (Eq. 4) does NOT depend
on P or Q at all -- only on M, kernel/stride/dilation geometry, and A. So
SWU (line-buffer) BRAM is IDENTICAL between FOLDING_UNFOLDED and
FOLDING_SERIAL -- folding trades away MVU compute/weight-memory resources,
never the sliding-window buffer. If SWU BRAM alone already exceeds budget,
no amount of folding fixes it -- only M (already minimal, =1), A (bit-width),
or the underlying kernel/stride/channel geometry can.

2026-09-17 REFRESH -- noActivation regime, formulae transcribed from the
FINN v1.0.0-alpha source checkout at <repo>/finn (src/finn/custom_op/
fpgadataflow/{matrixvectoractivation,vectorvectoractivation}.py, hls/
{matrixvectoractivation,vectorvectoractivation}_hls.py, rtl/
{matrixvectoractivation,vectorvectoractivation,convolutioninputgenerator,
thresholding}_rtl.py, util/basic.py). conv_cost_pe_simd now models, per
conv layer: MVAU/VVAU LUT (HLS noActivation formula; FINN's MVAU_rtl.
lut_estimation() is literally 0, so the HLS formula doubles as the RTL
proxy), DSP (HLS: P*Q*ceil((W+A)/48); RTL DSP48: ceil(P/lanes)*Q), weight
memory BRAM_18K (FINN's exact SDP aspect-ratio table) / URAM, the RTL
sliding-window unit (LUT=300, line-buffer BRAM and cycle count as a
function of the SWU's own SIMD), and the STANDALONE Thresholding_rtl node
that follows every MVAU/VVAU under noActivation=1 (empirical per-node cost,
now a real function of pe AND numSteps -- FINN's own LUTRAM-count estimator
contradicts the Vivado reports, see _thresholding_rtl_cost). Still not
covered: StreamingDataWidthConverters (need the successor layer's folding),
FIFOs, shell/interconnect. The old empirical "imbalance_luts" term is GONE:
it was measuring fused-threshold logic inside MVAU_hls (outputDataType=
UINTx on 172/173 calibration nodes), which noActivation=1 removes at the
source -- see analysis/hardware_calibration/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Folding = Literal["unfolded", "serial"]
FOLDING_UNFOLDED: Folding = "unfolded"
FOLDING_SERIAL: Folding = "serial"

# Empirical, BIT-WIDTH-DEPENDENT LUT/BRAM_18K derating factors (real_
# synthesis / this_model's_own_estimate). Originally (2026-08-25) a single
# flat factor calibrated against ONE real data point (S19 at uniform W8A8,
# hardware/results.csv's "s19_double_mid_8way_partitioned_ooc_synth_TOTAL"
# row, vs. this model evaluated at the real build's own resolved per-layer
# PE/SIMD -- hardware/outputs/s19_8way_partitioned_ooc_20260820_101224/
# final_hw_config.json, PE=SIMD=1 on effectively every MVAU node, i.e.
# avg_bits=8 uniform):
#     LUT:      830,689 real  vs. 100,996 this-model raw -> 8.225x
#     BRAM_18K:     906 real  vs.   1,495 this-model raw -> 0.606x
# A SECOND real data point (same day) proved the flat-factor assumption
# wrong: hardware/results.csv's "s19_hawq_block_partition_2_ooc_synth" row
# is a real OOC synthesis of partition_id=2 (down2 + stage2.0-2.4 +
# stage2.5.reduce.0 -- the largest of the 8-way S19 partitions, 23 real
# MVAU nodes) built at a REAL per-block HAWQ bit assignment AND its own
# REAL resolved per-layer folding (hardware/outputs/
# s19_hawq_block_partition_2_ooc_synth_20260824_220316/
# hawq_folding_config_partition2.json, PE=1 but SIMD in {4,6,8,12} --
# NOT FOLDING_SERIAL). IMPORTANT CORRECTION (2026-08-25, later same day):
# this anchor was FIRST computed wrong twice over -- (a) assuming that
# partition's real per-block bits were uniform W2A2 (they weren't: the real
# per-block assignment mixed W2/W4 weights and mostly W4A4/some W8 acts
# across down2/stage2.0-2.5, average (weight+act)/2 per layer, LUT-weighted
# across the partition's own 23 layers, is ~3.52, not 2), and (b) evaluating
# this model at the CURRENT (since-regenerated-by-this-session's-own-ILP-
# reruns) folding_block_s19.json instead of the REAL folding FINN actually
# built with (the hawq_folding_config_partition2.json file above). Both
# fixed by recomputing raw_total_lut/raw_total_bram18 directly from this
# model's own layer_cost_pe_simd(), fed the REAL per-block bits AND REAL
# per-layer (PE, SIMD) for all 23 layers, RAM_STYLE_BLOCK (BRAM_36K=0,
# URAM=0 in the real synthesis row, confirming no distributed/ultra RAM was
# used):
#     avg_bits=3.52 (LUT-weighted mean of (w+a)/2 across the partition's 23
#       real layers): LUT 22,436 real vs. 18,352 this-model raw -> 1.223x.
#       BRAM_18K 22 real vs. 171 this-model raw -> 0.129x.
# The derating factor still falls sharply at lower bit-width (LUT ~8.2x at
# avg_bits=8 down to ~1.2x at avg_bits=3.52; BRAM ~0.61x down to ~0.13x) --
# lines up with the real synthesis notes for that row: FINN's own resType
# heuristic packs narrow (2/4-bit) MACs into DSP48E2 slices instead of
# LUTs, and per-layer control/glue-logic overhead (the dominant term in why
# real LUT exceeds this closed-form model at all) doesn't scale down with
# bit-width the way raw arithmetic LUT usage does. A single flat factor
# (the original version of this module) applies the avg_bits=8-calibrated
# multiplier uniformly regardless of the ACTUAL bits chosen -- for a
# low-bit-heavy HAWQ assignment (which is exactly the common case: see
# ilp_search.py's own bit-search results, mostly 2/4-bit) that means
# systematically OVER-penalizing LUT/BRAM well beyond what real hardware
# would show, making the ILP overly conservative exactly where it matters
# most (a per-block search's whole point is to spend more bits only where
# sensitivity demands it -- if the cost model can't see that cheap bits are
# ACTUALLY cheap, it can't reward that choice).
#
# Model: linear interpolation of the derating factor between the two real
# anchors (avg_bits=3.52 and avg_bits=8 -- NOT [2, 8]: there is no real
# data point at avg_bits=2, see the correction above), using
# avg_bits = (weight_bits + act_bits) / 2 for whatever unit (stage/block/
# layer) is being costed, CLAMPED to [3.52, 8] -- this model has no basis
# to extrapolate below its lower real anchor (a genuinely all-2-bit
# assignment, avg_bits=2, would be extrapolating past the measured range,
# not interpolating; clamping to the avg_bits=3.52 factor is the
# conservative choice, i.e. probably still somewhat over-penalizing an
# even-lower-bit design, not under). Revisit the moment a real avg_bits<3.52
# or a real uniform-low-bit (e.g. true W2A2 across a whole real partition)
# data point exists.
#
# CAVEAT (same as before, now for TWO points instead of one): one
# architecture (S19), one folding regime each (the real build's own
# resolved PE/SIMD), a sum-of-independent-partitions build rather than a
# unified design. Two points fix the "is this even bit-width-dependent"
# question (clearly yes) but not the true curve shape -- treat interpolated
# values as a steering signal, not a guarantee.
_LUT_ANCHOR_BITS = (3.5245744425797163, 8)  # (avg_bits=3.52 real partition-2 HAWQ anchor, avg_bits=8 real whole-design anchor)
_LUT_ANCHOR_FACTORS = (22_436 / 18_352.4, 830_689 / 100_996)  # (~1.223 at avg_bits=3.52, ~8.225 at avg_bits=8)
_BRAM_ANCHOR_FACTORS = (22 / 171, 906 / 1_495)  # (~0.129 at avg_bits=3.52, ~0.606 at avg_bits=8)


def _interpolate_derating(avg_bits: float, anchor_factors: tuple[float, float]) -> float:
    """Linear interpolation between the avg_bits=3.52 and avg_bits=8
    real-synthesis anchor factors (see module comment above), clamped to
    [3.52, 8] -- this calibration has no basis to extrapolate below its
    lower real anchor (see module comment's correction note)."""
    lo_bits, hi_bits = _LUT_ANCHOR_BITS
    lo_factor, hi_factor = anchor_factors
    clamped = max(lo_bits, min(hi_bits, avg_bits))
    t = (clamped - lo_bits) / (hi_bits - lo_bits)
    return lo_factor + t * (hi_factor - lo_factor)


def calibrated_lut(
    raw_total_lut: float, weight_bits: float, act_bits: float, force_dsp: bool = False, lut_mult: bool = False,
) -> float:
    """This model's own raw total_lut, corrected by a derating factor
    interpolated between the avg_bits=3.52/avg_bits=8 real-synthesis
    anchors at avg_bits=(weight_bits+act_bits)/2 -- see the module-level
    comment above for the calibration this is based on and its real scope
    limits.

    force_dsp=True switches to the FORCED-DSP regime's own flat factor
    (see _FORCED_DSP_LUT_FACTOR below) instead -- default False preserves
    this function's exact prior behavior for every existing caller.

    lut_mult=True (2026-09-18, additive/opt-in, default False preserves prior
    behavior exactly) is for the "hls_lut_noact0" resource variant
    specifically: conv_cost_pe_simd's own mvu_lut already applies a REAL,
    node-type-specific derate for this regime (_HLS_MVU_LUT_MULT_DERATE, fit
    on real MVAU_hls+LUT-mult per-node ground truth) -- applying THIS
    function's avg_bits table on top would double-derate with an unrelated,
    wrong-direction correction: at avg_bits=8 it multiplies by 8.225x (fit on
    an older, different real build, before the noActivation-choice variant
    framework existed at all), which taking one real probe node as an
    example (MVAU_hls_0, raw=1338, real=953) would inflate the ALREADY-
    derated 991 prediction to 11,008 -- 11.5x too high, not a correction.
    So lut_mult=True skips this table entirely (identity), the same way
    force_dsp=True already does, leaving _HLS_MVU_LUT_MULT_DERATE as the
    ONLY derate applied for this regime. (The old "untested for this regime"
    caveat this docstring used to carry is now resolved: it WAS wrong, hence
    this bypass, not just uncalibrated.)"""
    if force_dsp or lut_mult:
        return raw_total_lut * _FORCED_DSP_LUT_FACTOR if force_dsp else raw_total_lut
    avg_bits = (weight_bits + act_bits) / 2
    return raw_total_lut * _interpolate_derating(avg_bits, _LUT_ANCHOR_FACTORS)


def calibrated_bram18k(raw_total_bram18k: float, weight_bits: float, act_bits: float, force_dsp: bool = False) -> float:
    """This model's own raw BRAM_18K total, corrected the same way as
    calibrated_lut -- see that function's and the module-level comment."""
    if force_dsp:
        return raw_total_bram18k * _FORCED_DSP_BRAM_FACTOR
    avg_bits = (weight_bits + act_bits) / 2
    return raw_total_bram18k * _interpolate_derating(avg_bits, _BRAM_ANCHOR_FACTORS)


# ---------------------------------------------------------------------------
# FORCED-DSP regime (2026-09-05): two real S12 8-way-partitioned OOC builds,
# both forcing DSP on every MVAU/VVAU node -- a different regime from the
# auto-resType anchors above (see compression/hawq/fit_forced_dsp_derating.py
# for the full fitting script and provenance; plan:
# C:\Users\win32\.claude\plans\nested-singing-flurry.md).
#
# Per-PARTITION real ground truth (8 partitions x 2 builds = 16 points) was
# fit two ways:
#   (a) ratio real/raw vs. avg_bits -- broke down badly for partition 0 (the
#       single initial.conv-only partition in both builds: raw_lut=937/1360,
#       real_lut=8909/4297, i.e. 9.5x/3.2x, vs. every other partition's
#       0.98x-1.86x). NOT because partition 0 is expensive in absolute terms
#       (it's the cheapest partition in both builds) -- because a roughly
#       FIXED per-partition synthesis overhead (I/O shims, FIFOs, clock/reset
#       infra) doesn't shrink with the partition's own logic, so dividing it
#       by an equally tiny raw baseline produces a huge ratio. This is a
#       property of a pure-multiplicative model, not of partition 0.
#   (b) real_lut = a*raw_lut + b (affine, all 16 points, INCLUDING partition
#       0): R^2=0.916 -- partition 0's residual falls in line with everyone
#       else's once the fixed intercept exists to absorb exactly that
#       per-partition overhead. Confirms (a)'s diagnosis directly. Adding
#       total PE/total SIMD as EXTRA regressors alongside raw_lut barely
#       moves R^2 (0.916->0.926, with a sign-flipped SIMD coefficient --
#       collinearity noise from 4 parameters on 16 points) since raw_lut
#       already IS the PE*SIMD*bits combination (see conv_cost_pe_simd's own
#       mvu_lut formula); PE/SIMD ALONE, without raw_lut, fits WORSE
#       (R^2=0.817) than raw_lut alone. Regressing on the already-combined
#       raw estimate is strictly better than re-splitting it into PE/SIMD
#       terms with too little data to support the extra parameters.
#   BRAM's affine fit is much noisier (R^2=0.462) -- real BRAM_18K counts per
#       partition are tiny (5-36), dominated by integer-rounding noise at
#       that scale; treat BRAM_18K forced-DSP estimates as rougher than LUT's.
#
# The affine model's intercept is a genuine PER-PARTITION fixed cost, but
# calibrated_lut/calibrated_bram18k are called per-LAYER, inside the ILP,
# before partition boundaries exist -- so the affine form can't be applied
# per-layer without over-counting the intercept once per layer instead of
# once per partition. Two separate uses instead:
#
#   1. Per-layer factor for the ILP's own search (relative cost signal
#      across candidate bit assignments): _FORCED_DSP_LUT_FACTOR/
#      _FORCED_DSP_BRAM_FACTOR below, a FLAT mean factor (not a function of
#      avg_bits) computed over the 14 non-degenerate partitions (excluding
#      each build's own partition 0 -- a single-layer partition's ratio is
#      inherently unreliable for exactly the fixed-overhead reason above,
#      not a real per-layer bits effect). Flat, not sloped, because the real
#      avg_bits range across those 14 partitions is only [4.0, 5.15] -- S12's
#      min4 folding convention keeps per-layer bits tightly clustered there,
#      too narrow a range to support a real slope estimate.
#   2. Affine TOTAL-cost check (_forced_dsp_lut_total/_forced_dsp_bram_total
#      below) for validating a CONCRETE partitioned plan (n_partitions known)
#      against a hard cap post-hoc -- n_partitions*b + a*raw_total.
# Historical calibration, PRESERVED for provenance/reuse -- both real builds
# behind these numbers are 12_separable_dense_relu_min4 (S12 SEPARABLE) ONLY
# (see fit_forced_dsp_derating.py's BUILDS dict). Confirmed via real FINN
# source read (2026-09-16, see repo memory finn_calibrated_8way_build_status.md's
# "LUT model mismatch: CONFIRMED root cause" section) that this flat/affine
# correction was silently absorbing a whole missing structural term (real
# FINN's addertree_luts/acc_luts, the latter MW-dependent) that mvu_lut now
# models directly below -- a factor fit only on separable-architecture
# (small-MW) data under-corrects for dense (non-separable, large-MW)
# architectures (confirmed empirically below: the dense-specific refit's own
# flat factor, 4.46x, is ~3.5x larger than separable's 1.26x, even AFTER
# mvu_lut's structural fix -- see fit_forced_dsp_derating_s12_dense.py's own
# per-layer folding comparison: the real dense build's ILP chose far LESS
# folding (e.g. stage2.0.conv real SIMD=24 vs separable's matched-slot
# SIMD=4-6, and this session's re-solved alpha=0.25 output pushes that same
# dense layer to SIMD=72, i.e. fully UNFOLDED) than separable's did, so a
# large chunk of the real gap is a folding-choice/addertree_luts-scaling
# effect, not pure architecture). Kept under an architecture-specific name
# rather than deleted: still a valid anchor for a from-scratch check against
# the S12 SEPARABLE geometry specifically.
_S12_SEPARABLE_DSP_FORCED_LUT_FACTOR = 1.261221175430389   # mean lut_factor, 14 partitions (both builds, partition 0 excluded), avg_bits in [4.0, 5.15]
_S12_SEPARABLE_DSP_FORCED_BRAM_FACTOR = 0.19434811541501412  # mean bram_factor, same 14 partitions
_S12_SEPARABLE_DSP_FORCED_LUT_AFFINE = (0.947632331261822, 4208.818846016495)   # (a, b): real_lut = a*raw_lut + b, all 16 partitions, R^2=0.916
_S12_SEPARABLE_DSP_FORCED_BRAM_AFFINE = (0.08739251550203067, 8.135659131252611)  # (a, b): real_bram18 = a*raw_bram18 + b, all 16 partitions, R^2=0.462

# S12 DENSE (non-separable, SEPARABLE_DILATED=False) refit (2026-09-15) --
# see fit_forced_dsp_derating_s12_dense.py and its own output
# compression/hawq/artifacts/forced_dsp_derating_fit_s12_dense.json. ONE real
# build (quantEnet_12_dense_relu_warmstart150ep_alpha025_finn_calibrated_
# int8, 8 partitions, no second build to pool against unlike separable's 16
# points) -- weaker statistically than the separable fit above. avg_bits
# range [5.003, 7.284] (this build's own {4,6,8}-candidate joint search, vs
# separable's tighter min4 [4.0, 5.15]).
#   - Flat lut_factor: mean=4.4639 (RMSE=1.6317) over all 8 partitions
#     (unlike separable, dense's own partition 0 -- avg_bits=6.0,
#     factor=5.24 -- is NOT a fixed-overhead outlier the way separable's was
#     (9.5x/3.2x there vs the 2.78-6.41 range here for the other 7), so
#     nothing excluded).
#   - Affine lut fit: a=-1.1204, b=10.9734, R^2=0.268 -- WEAK and
#     WRONG-SIGNED (more bits -> less LUT makes no physical sense) at this
#     sample size; NOT used for the active _FORCED_DSP_LUT_AFFINE below
#     (left at identity) -- an 8-point regression isn't enough evidence to
#     override the post-hoc total-cost check with a spurious slope. Stored
#     here for provenance only.
#   - Flat bram_factor: mean=0.4598 (RMSE=0.1245) -- REAL BRAM_18K usage is
#     UNDER its own raw prediction here (factor <1), the opposite direction
#     from LUT (which needs a >1 correction) -- an asymmetry worth
#     remembering, not a typo.
_S12_DENSE_DSP_FORCED_LUT_FACTOR = 4.4639          # flat mean, all 8 real partitions, 1 build
_S12_DENSE_DSP_FORCED_BRAM_FACTOR = 0.4598         # flat mean, same 8 partitions
_S12_DENSE_DSP_FORCED_LUT_AFFINE = (-1.1204, 10.9734)   # (a, b): lut_factor = a*avg_bits + b, R^2=0.268 -- weak, NOT applied by default, provenance only
_S12_DENSE_DSP_FORCED_BRAM_AFFINE = (0.1131, -0.1973)   # (a, b): bram_factor = a*avg_bits + b, R^2=0.468 -- also weak at n=8, provenance only

# ACTIVE default (2026-09-15): the S12 DENSE flat factor above, not identity.
# mvu_lut's addertree_luts/acc_luts structural terms (2026-09-16 fix) closed
# part of the gap but a direct cross-check (this session: re-pricing the
# REAL as-synthesized dense partitions under the fixed formula) still showed
# a real/raw ratio of ~4.5-7x -- the structural fix alone was NOT sufficient
# for dense geometry, so a non-trivial derating is back, sourced from the
# dense-specific refit rather than reused from separable's (confirmed wrong
# for dense: separable's own factor, 1.26x, is ~3.5x too small here).
# CAVEAT: this is a SINGLE global slot, not dispatched by architecture --
# applying it to a future SEPARABLE-geometry force_dsp estimate would
# OVER-correct (separable only needs ~1.26x, per _S12_SEPARABLE_... above).
# No per-geometry selector exists yet; whoever reuses calibrated_lut(...,
# force_dsp=True) for a non-S12-dense architecture should swap this back to
# 1.0 or to the relevant _S12_*_DSP_FORCED_LUT_FACTOR by hand until a real
# dispatch mechanism is built. The AFFINE constants (used only by
# forced_dsp_lut_total/forced_dsp_bram_total, not by the per-layer ILP
# search) are LEFT AT IDENTITY -- dense's own affine fit above is too weak
# (R^2=0.268, wrong-signed) to trust for that post-hoc hard-cap check.
#
# 2026-09-17: BOTH flat factors reset to IDENTITY. Every S12 factor above
# was fit on builds whose MVAU_hls nodes carried FUSED thresholds
# (outputDataType=UINTx on 172/173 calibration nodes -- the very thing that
# made FINN pick HLS over RTL and produced the 4-6x LUT blow-up). The model
# now assumes noActivation=1 + MVAU_rtl and prices the standalone
# Thresholding_rtl explicitly (conv_cost_pe_simd), so those factors no
# longer describe the regime being estimated; the six real noActivation/RTL
# probes in hardware/results.csv sit at 1.1-1.7x of the raw model. Refit
# against the RTL production rebuild when it lands; the S12 constants are
# kept above for provenance only.
_FORCED_DSP_LUT_FACTOR = 1.0
_FORCED_DSP_BRAM_FACTOR = 1.0
_FORCED_DSP_LUT_AFFINE = (1.0, 0.0)
_FORCED_DSP_BRAM_AFFINE = (1.0, 0.0)


def forced_dsp_lut_total(raw_total_lut: float, n_partitions: int) -> float:
    """Affine total-cost check for a CONCRETE forced-DSP partitioned plan
    (n_partitions known) -- n_partitions*b + a*raw_total, fit against all 16
    real S12 partition points including partition 0 (see module comment
    above). NOT for use inside the per-layer ILP search (n_partitions isn't
    known there) -- use calibrated_lut(..., force_dsp=True) for that."""
    a, b = _FORCED_DSP_LUT_AFFINE
    return a * raw_total_lut + n_partitions * b


def forced_dsp_bram_total(raw_total_bram18k: float, n_partitions: int) -> float:
    """Same as forced_dsp_lut_total, for BRAM_18K -- see that function's
    docstring. BRAM's affine fit is noisier (R^2=0.462, see module comment)."""
    a, b = _FORCED_DSP_BRAM_AFFINE
    return a * raw_total_bram18k + n_partitions * b

# Weight-memory FPGA resource choice for the MVU's weight tile (FINN's own
# "ram_style" nodeattr, see matrixvectoractivation.py: block=BRAM, ultra=
# URAM; mutually exclusive, real FINN's bram_estimation()/uram_estimation()
# return 0 for the style not selected). "distributed" (LUTRAM) is not
# modeled here -- real FINN's lut_estimation() adds an extra c2 LUT term
# for that style only, which this closed-form model doesn't (yet) carry;
# not needed for the block-vs-ultra BRAM/URAM trade this file supports.
RamStyle = Literal["block", "ultra"]
RAM_STYLE_BLOCK: RamStyle = "block"
RAM_STYLE_ULTRA: RamStyle = "ultra"

# Which FINN backend the MVAU specializes to (step_specialize_layers). RTL is
# what FINN picks by itself for an MVAU with noActivation=1, SIGNED weights
# (>= 2 bit) and <= 8-bit operands on DSP48E2 (_mvu_rtl_possible); anything
# with a fused activation or UNSIGNED weights (this repo's UINT7/UINT3
# weight nodes) stays HLS. VVAU_rtl is Versal-only, so depthwise layers are
# always HLS on xczu7ev regardless of this setting.
ImplStyle = Literal["hls", "rtl"]
IMPL_STYLE_HLS: ImplStyle = "hls"
IMPL_STYLE_RTL: ImplStyle = "rtl"

# Standalone Thresholding_rtl, EMPIRICAL per-node cost, now a real function
# of BOTH pe AND numSteps (2026-09-17 refit -- see below; the previous
# pe-only/flat-BRAM version is kept only in git history). numSteps = number
# of real threshold values FINN's Thresholding_rtl actually stores =
# 2**output_bits - 1 (this is a direct structural fact of threshold-based
# quantization -- to sort a value into one of 2^B output levels you compare
# against 2^B-1 sorted thresholds, see thresholding.sv's own binary-search
# pipeline -- NOT a fitted relationship). output_bits here is THIS layer's
# own act_bits (the ILP's y[layer,w,a] choice) as the best available proxy
# for the real outputDataType bitwidth Thresholding_rtl actually stores --
# same approximation already used for the fused-threshold case elsewhere in
# this file (no separate output-precision axis exists), since a standalone
# node's real successor-facing output type isn't threaded into this
# function's own (layer, weight_bits, act_bits) call signature.
#
# Refit 2026-09-17 on 168 real Thresholding_rtl nodes (hardware/datasets/
# mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_alpha025_rtl_
# mvau_noact1_extended.csv -- the REAL S12-dense-warmstart alpha=0.25 8-way
# production build), TRAIN split; hardware/datasets/mvau_variant_matrix_
# dataset_extended.csv's noact1_auto_* rows (n=18, but PE=1/numSteps=255
# for every single row -- zero variance in either axis, so this slice
# cannot validate a PE or numSteps DEPENDENCE, only the overall level at
# that one point) held out as TEST, per finn_milp.py's own S12-dense-
# warmstart no-op-regression convention -- not a proper held-out validation
# of this specific relationship, flagged honestly rather than glossed over.
#
# The OLD flat/pe-only model was confirmed WRONG, not just imprecise: real
# BRAM18 at PE=1/numSteps=15 (4-bit output) averages 0.05 (n=20) -- the old
# flat _THR_RTL_BRAM18_PER_NODE=3.6835 constant overshoots that ~74x -- while
# PE=8/numSteps=255 (8-bit) averages 19.0 -- the same flat constant
# undershoots THAT ~5x. Both LUT and BRAM needed numSteps added, not just PE:
#   LUT:   real_LUT     ~= PE * (_THR_RTL_LUT_BASE_PER_PE + _THR_RTL_LUT_PER_NUMSTEP_PE * numSteps)
#          -- PE-only R^2=0.950; adding the numSteps term -> R^2=0.960.
#          Physically: a fixed per-PE-lane comparator/control cost
#          (independent of how many thresholds) plus a per-PE-lane term that
#          scales with numSteps (each PE lane's own LUTRAM-packed threshold
#          set). OLS (through origin on both terms; no free intercept, same
#          convention as every other constant in this file).
#   BRAM18: real_BRAM18 ~= _THR_RTL_BRAM18_PER_PE_NUMSTEP * PE * numSteps
#          -- PE-only R^2=0.707; numSteps-only R^2=0.232; the PURE
#          interaction term alone (no separate PE or numSteps terms) gets
#          R^2=0.812, actually BEATING every free-intercept multi-term model
#          tried (which had unstable, sign-flipping coefficients from PE/
#          numSteps collinearity in this dataset -- 4 PE values x 3 numSteps
#          values, not a full factorial). A single clean multiplicative
#          constant generalizes more honestly than an overfit 3-4 parameter
#          model here.
_THR_RTL_LUT_BASE_PER_PE = 70.6827        # fixed per-PE-lane cost, independent of numSteps
_THR_RTL_LUT_PER_NUMSTEP_PE = 0.1065      # additional per-PE-lane cost, scales with numSteps
_THR_RTL_BRAM18_PER_PE_NUMSTEP = 0.011444  # pure PE*numSteps interaction, no separate PE/numSteps terms

# _THR_RTL_URAM_PER_PE_NUMSTEP: DERIVED, not fit -- real_URAM==0 for all
# 168/168 Thresholding_rtl nodes in the training data (depth_trigger_uram=0
# throughout, FINN's own per-stage min-waste primitive selection -- see
# finn.util.basic.mem_primitives_versal/get_memutil_alternatives, used
# UNCONDITIONALLY regardless of target part despite the "versal" name (no
# other primitives table exists in FINN, and this exact estimator is what
# already produced the real_BRAM18 data the sibling constant above was
# fit against, on our real non-Versal xczu7ev -- so whatever inaccuracy
# that naming implies is already absorbed into that fit, not a new
# problem here) -- never once favored URAM for this network's own
# geometries. So there is no real per-node ratio to fit; this constant is
# instead SCALED from the real, validated BRAM18 constant by the fixed,
# device-family-independent capacity ratio of the two real primitives:
# BRAM18 = 18Kib = 18,432 bits (the "36x512" shape in mem_primitives_versal,
# 36*512) vs URAM288 = 288Kib = 294,912 bits (the native "72x4096" shape,
# 72*4096) -- exactly 16x denser per block. This assumes forcing a
# Thresholding node's memory into URAM preserves roughly the same relative
# packing efficiency BRAM currently achieves, just in a 16x-bigger unit --
# a reasonable structural assumption, not a validated one (zero real
# Thresholding+URAM hardware data exists anywhere yet). Was briefly a real,
# free per-layer BRAM-vs-URAM choice in finn_milp.py's ILP (2026-09-17/18),
# to trade BRAM pressure onto idle URAM -- RETIRED 2026-09-18: a real
# attempted Vivado synthesis of Thresholding_rtl with ram_style="ultra"
# FAILS outright. URAM288 cannot be used as ROM, and Thresholding's memory
# is exactly that: a compile-time-constant lookup table (the threshold
# values), never written at runtime -- URAM lacks the INIT-file-based ROM
# initialization mechanism BRAM18/36 primitives have. finn_milp.py now
# hard-fixes thr_ram_style="block" unconditionally; this constant and the
# ram_style="ultra" branch below are KEPT (not deleted) for provenance and
# any future diagnostic/what-if use, but are no longer reachable from any
# live ILP run.
_THR_RTL_URAM_PER_PE_NUMSTEP = _THR_RTL_BRAM18_PER_PE_NUMSTEP * (18_432 / 294_912)  # = 0.0007153

# _THR_RTL_LUTRAM_PER_PE_NUMSTEP (2026-09-18): DERIVED, not fit, same method
# as the URAM constant above but landing somewhere genuinely usable this
# time. Unlike ram_style="ultra" (real Vivado synthesis FAILS outright --
# URAM cannot be ROM, see finn_milp.py's RAM_STYLES comment), "distributed"
# IS structurally real for Thresholding_rtl: thresholding.sv's own RAM_STYLE
# localparam has a genuine "auto"/"distributed" choice (confirmed via direct
# .sv source read, not just the Python cost-estimator), forceable per-node
# by setting depth_trigger_bram above the node's own real depth (see that
# same RAM_STYLES comment for exactly how). It's just never been exercised
# in any real build (every real node so far used "auto", i.e. Vivado's own
# choice, which always landed on BRAM for the geometries tested -- 0/240
# real Thresholding_rtl nodes across both datasets ever showed nonzero
# real_LUTRAM). So: no real per-node ratio to fit here either, same
# situation as URAM. Scaled from the same real, validated BRAM18 constant,
# but by the real capacity ratio to FINN's own LUTRAM primitive shape
# instead of URAM288's: finn.util.basic.mem_primitives_versal defines
# "LUTRAM": (1, 64) -- width=1 bit, depth=64 words, i.e. 64 bits per
# primitive (this is a real, standard Xilinx shape: one LUT6 configured as
# RAM64X1S distributed RAM, not an invented number). BRAM18 = 18,432 bits
# per block is 288x denser than one 64-bit LUTRAM primitive, so the same
# structural assumption as the URAM derivation (forcing this memory into a
# different primitive preserves roughly BRAM's own real packing efficiency,
# just in a much SMALLER unit this time, hence MORE primitives, not fewer)
# gives a MUCH larger per-(pe,numStep) coefficient than URAM's -- and
# correctly so: for this project's real numSteps range (commonly 255),
# distributed genuinely IS a bad deal (predicts ~840 LUTs for a numSteps=255
# node needing only ~3 BRAM18 blocks), which is exactly WHY Vivado's own
# "auto" heuristic has never once picked it in any real build -- the
# derivation reproduces that real-world avoidance rather than contradicting
# it. It only becomes competitive at small numSteps, where BRAM's own fixed
# per-block overhead dominates -- exactly the resource-pressure tradeoff an
# ILP is suited to evaluate per layer, unlike a fixed local rule.
_THR_RTL_LUTRAM_PER_PE_NUMSTEP = _THR_RTL_BRAM18_PER_PE_NUMSTEP * (18_432 / 64)  # = 3.295872


@dataclass
class LayerGeometry:
    """One Conv2d/ConvTranspose2d/MaxPool2d layer's shape, in the same
    convention as enet_finn_fully_unfolded_M1_per_layer.csv's columns."""
    op_type: str  # "Conv2d" | "ConvTranspose2d" | "MaxPool2d"
    name: str
    stage: str
    cin: int
    hin: int
    win: int
    cout: int
    hout: int
    wout: int
    kh: int
    kw: int
    sh: int
    sw: int
    dh: int = 1
    dw: int = 1
    groups: int = 1
    # groups=1 (default) is byte-for-byte identical to this field not
    # existing. groups=cin=cout is a true depthwise conv (see ENet.py's
    # DSCNoProjectionBottleneck/RegularBottleneck's use_dsc branch): each
    # output channel reduces over only cin/groups input channels, NOT the
    # full cin -- previously silently treated as a DENSE conv (Q=cin*kh*kw
    # instead of the real (cin/groups)*kh*kw), overstating LUT/BRAM/PE/SIMD
    # by a factor of ~groups for every depthwise layer in any DSC/
    # dsc_no_projection architecture (S8/S10/S13/S15/S16/S19-DSC variants,
    # 22_dsc_projected). Does NOT affect swu_bram18 (the sliding-window
    # buffer still holds all cin input channels' worth of pixels regardless
    # of grouping) or any architecture that never sets groups>1 (e.g. the
    # 26_9_w24_s14w12_nonneg_block family, which uses separable_dilated's
    # (k,1)+(1,k) DENSE factoring, not grouped convs -- unaffected by this
    # fix either way).


def is_depthwise(layer: LayerGeometry) -> bool:
    """True for a real grouped-depthwise Conv2d (groups>1) -- the shape
    that becomes a FINN VVAU (Vector-Vector-Activation-Unit) node instead
    of an MVAU, with its own PE/SIMD folding domain (see
    swu_max_simd_depthwise below) and its own preceding SWU/FMPadding
    coupling constraint (compression/hawq/folding_ilp.py's
    solve_folding_nodewise). This repo only ever constructs FULLY
    depthwise convs (groups == cin == cout, one filter per channel) via
    ENet.py's DSCNoProjectionBottleneck / RegularBottleneck(use_dsc=True)
    -- assert that shape defensively rather than silently mis-handling a
    partial-group conv, which this file's VVAU domain helpers don't
    support and no config in this repo ever produces."""
    if layer.op_type != "Conv2d" or layer.groups <= 1:
        return False
    assert layer.groups == layer.cin == layer.cout, (
        f"{layer.name}: partial-group conv (groups={layer.groups}, cin={layer.cin}, "
        f"cout={layer.cout}) is not a supported VVAU shape -- only fully depthwise "
        f"(groups==cin==cout) convs are modeled here."
    )
    return True


def swu_max_simd_depthwise(layer: LayerGeometry) -> int:
    """The preceding ConvolutionInputGenerator's (SWU's) own SIMD must
    divide IFMChannels -- for a depthwise layer that's cin==cout==groups
    (the channel count), NOT cin*kh*kw (that's the MVAU-SIMD constraint for
    a DENSE conv, a different axis -- see hardware/finn_native_cost_
    estimator.py's own module docstring for the real-FINN source-derived
    note on this mismatch). Deliberately the SAME domain as this layer's
    own VVAU PE (max_pe(layer)==layer.cout) -- solve_folding_nodewise's
    coupling constraint requires an SWU SIMD choice and a VVAU PE choice to
    be able to agree exactly."""
    return layer.cout


MEM_MODE_DECOUPLED = "internal_decoupled"  # real FINN v0.10.1 mem_mode nodeattr
                                            # value for MVAU/VVAU nodes whose
                                            # weights are streamed from BRAM/
                                            # URAM rather than embedded as LUT/
                                            # FF constants (confirmed against a
                                            # real generated auto_folding_
                                            # config.json -- the informal name
                                            # "decoupled" is this version's
                                            # stale/pre-rename name for it).


def _k_eff(kh: int, dh: int) -> int:
    return (kh - 1) * dh + 1


def _pq_for_folding(layer: LayerGeometry, folding: Folding) -> tuple[int, int]:
    if folding == FOLDING_UNFOLDED:
        return (layer.cin // layer.groups) * layer.kh * layer.kw, layer.cout  # Q, P: whole reduction + all output channels/cycle
    if folding == FOLDING_SERIAL:
        return 1, 1  # Q, P: one reduction element, one output channel/cycle -- minimum resource, maximum latency
    raise ValueError(f"Unknown folding {folding!r}, expected one of {FOLDING_UNFOLDED!r}/{FOLDING_SERIAL!r}.")


def max_pe(layer: LayerGeometry) -> int:
    return layer.cout


def max_simd(layer: LayerGeometry) -> int:
    return (layer.cin // layer.groups) * layer.kh * layer.kw


def divisors(n: int) -> list[int]:
    """FINN's real folding constraint: PE must evenly divide C_out, SIMD
    must evenly divide C_in*K_h*K_w -- ragged folding isn't a clean native
    MVAU config (some FINN versions pad to support it; not modeled here,
    same "closed-form estimate, not an actual FINN build" scope as the rest
    of this file)."""
    return [d for d in range(1, n + 1) if n % d == 0]


def _finn_wm_bram18(omega: float, mem_width: int, depthwise: bool) -> int:
    """FINN MVAU.bram_estimation()/VVAU.bram_estimation() (base classes,
    shared by the _hls and _rtl backends), ram_style="block": RAMB18 count
    for the decoupled weight memory, omega words deep x mem_width bits wide,
    using FINN's SDP-mode aspect-ratio table (UG573 Table 1-10). The VVAU
    variant assumes slightly narrower usable widths (8/16/32 vs 9/18/36)."""
    w9, w18, w36 = (8, 16, 32) if depthwise else (9, 18, 36)
    if mem_width == 1:
        return math.ceil(omega / 16384)
    if mem_width == 2:
        return math.ceil(omega / 8192)
    if mem_width <= 4:
        return math.ceil(omega / 4096) * math.ceil(mem_width / 4)
    if mem_width <= 9:
        return math.ceil(omega / 2048) * math.ceil(mem_width / w9)
    if mem_width <= 18 or omega > 512:
        return math.ceil(omega / 1024) * math.ceil(mem_width / w18)
    return math.ceil(omega / 512) * math.ceil(mem_width / w36)


def _finn_buffer_bram18(buffer_width: int, buffer_depth: int) -> int:
    """FINN ConvolutionInputGenerator_rtl.bram_estimation()'s RAMB18 count
    for ONE line buffer (ram_style block/auto): aspect ratio chosen by depth,
    cascaded past 16384 words, with FINN's own remainder-cascade saving."""
    def ram_width_for(depth: int) -> int:
        for limit, width in ((512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2)):
            if depth <= limit:
                return width
        return 1

    cascade_depth = math.ceil(buffer_depth / 16384)
    cascade_width = math.ceil(buffer_width / ram_width_for(buffer_depth))
    savings = 0
    if buffer_depth > 16384:
        savings = cascade_width - math.ceil(buffer_width / ram_width_for(buffer_depth % 16384))
    return int(cascade_depth * cascade_width - savings)


def _finn_buffer_uram18(buffer_width: int, buffer_depth: int) -> int:
    """FINN ConvolutionInputGenerator_rtl.uram_estimation()'s URAM288 count
    for ONE line buffer (ram_style="ultra"): fixed 4096-deep x 72-wide
    aspect ratio (real URAM288 shape, same constants MVAU's own
    uram_estimation() uses for weight memory -- see conv_cost_pe_simd's own
    wm_uram18 computation) -- no aspect-ratio table or cascade-remainder
    saving the way BRAM has, since URAM only comes in this one shape.
    UNCALIBRATED: every real SWU node checked in this repo's calibration
    data (43/43) used ram_style="distributed" -- there is no real
    ram_style="ultra" SWU ground truth yet, so this is a direct FINN-source
    transcription, not a fitted/verified formula (same status as
    _finn_buffer_bram18, which is equally untested against real "block"
    SWU data -- see _finn_swu's own docstring)."""
    cascade_depth = math.ceil(buffer_depth / 4096)
    cascade_width = math.ceil(buffer_width / 72)
    return int(cascade_depth * cascade_width)


_SWU_LUT_DERATE = 0.755  # provisional, OLS-through-origin fit, n=43 real ConvolutionInputGenerator_rtl
                          # nodes (hardware/mvau_swu_threshold_calibration_dataset.csv) -- real Vivado
                          # LUT comes in 51-104% of this formula's estimate (mean 70%), same direction
                          # as _RTL_MVU_LUT_DERATE (real RTL synthesis under FINN's own analytical
                          # estimate). Refit once the full production rebuild lands.


def _finn_swu(
    layer: LayerGeometry, act_bits: int, simd_swu: int, depthwise: bool, parallel_window: bool,
    ram_style: str = "distributed",
) -> tuple[int, int, int, int]:
    """FINN ConvolutionInputGenerator_rtl (the sliding-window unit feeding
    this layer's MVAU/VVAU): (swu_lut, swu_bram18, swu_uram18, swu_cycles).

    A 1x1 kernel gets NO SWU NODE AT ALL -- confirmed structurally, not
    numerically: across every real node in hardware/mvau_swu_threshold_
    calibration_dataset.csv (production S12-separable partitions + the
    int6_pemh probe, whose exact per-layer kernel sizes are known), MVAU
    (MH,MW) pairs where MW==IFMChannels (kh=kw=1, e.g. every ENet
    bottleneck's 1x1 reduce/expand) are NEVER immediately preceded by a
    ConvolutionInputGenerator_rtl node (38/38 occurrences); pairs with a
    real kernel area (MW a multiple of IFMChannels, e.g. 3x1/1x3/3x3)
    ALWAYS are (35/35). A 1x1 conv reads one input pixel per output pixel
    with nothing to buffer, so FINN's own InferConvInpGen transform simply
    doesn't insert the node -- this is a graph-structure fact, not
    something get_buffer_depth()'s formula (which would still return a
    small nonzero value at kh=kw=1) would tell you.

    For a real kernel (kh,kw not both 1): BRAM and cycles depend on the
    SWU's OWN SIMD through channel_factor = IFMChannels/SIMD -- a small
    SIMD means a DEEP, NARROW line buffer (worse RAMB18 aspect ratio) and
    proportionally more cycles per output row -- this is where FINN's
    "unfold SIMD before PE" guidance is actually grounded. impl_style
    follows select_impl_style(): "parallel" for parallel_window=1 (window
    emitted whole, cycles = number of input words + 2), else "default"
    (get_buffer_depth()/get_exp_cycles()'s 2D branch). The 1D-input branch
    (ifm_dim_h==1 or ifm_dim_w==1) is not reproduced -- no layer in this
    repo has a 1-pixel-high/wide input. No real parallel_window=1 node
    exists yet in the calibration data (all 43 are impl_style="default")
    to confirm get_buffer_depth()'s own "parallel" formula (used here) --
    NOT bram_estimation()'s separate inline "parallel" buffer_depth
    (kernel_width/(win-kernel_width) form, used only for THAT function's
    own BRAM sizing, and easy to confuse with this one -- an earlier
    version of this function did exactly that).

    ram_style (2026-09-17 addition, additive/opt-in, default "distributed"
    preserves this function's exact prior behavior for every existing
    caller): FINN's ConvolutionInputGenerator_rtl.lut_estimation() adds a
    LUTRAM term (buffer_width*ceil(buffer_depth/38)) ONLY for ram_style=
    "distributed" (the default); bram_estimation() only counts BRAM (via
    _finn_buffer_bram18) for "block"/"auto"; uram_estimation() only counts
    URAM (via _finn_buffer_uram18) for "ultra" -- all three mutually
    exclusive, transcribed directly from FINN's own three functions. A
    fourth value, "auto_efficient" (NOT a real FINN ram_style -- resolved
    internally, before any of the three real values are ever set), computes
    BOTH _finn_buffer_bram18 and _finn_buffer_uram18 for this exact
    (buffer_width, buffer_depth) and deterministically keeps whichever needs
    FEWER physical blocks (ties go to block) -- this is what "does this
    buffer's own geometry favor URAM" means concretely: not real FINN's own
    "auto" (which never considers URAM at all, same blind spot as MVAU's own
    ram_style="auto"), but a genuine per-buffer efficiency comparison. This
    is what finn_milp.py's ILP passed for every layer BRIEFLY (2026-09-17),
    before being retired the very next day (2026-09-18): by then 48/48 real
    SWU nodes across BOTH real datasets (the original 43-node calibration
    set AND the mvau_variant_matrix probe) had used ram_style="distributed"
    with real_BRAM18==real_URAM==0, REGARDLESS of what ram_style the paired
    MVAU's own weight memory used -- and for the actual deployed geometry
    this "auto_efficient" mode predicted a NON-trivial 41 URAM blocks (42.7%
    of the 96-block budget), a real, consequential-sized claim with zero
    supporting evidence, not a rounding error. finn_milp.py now hard-fixes
    SWU to swu_ram_style="distributed" instead (same treatment MVAU's own
    "ultra" request already got, for the identical reason). This mode is
    KEPT here (not deleted) for provenance and any future diagnostic/
    what-if use, but is no longer reachable from any live ILP run. Both
    _finn_buffer_bram18/_finn_buffer_uram18 remain direct FINN-source
    transcriptions with NO real "block"/"ultra" SWU ground truth ANYWHERE
    (see their own docstrings) -- this auto-pick was only ever as
    trustworthy as those two formulas are, which is to say: plausible, and
    now empirically shown to not match real hardware."""
    kh, kw, dh, dw, sh, sw = layer.kh, layer.kw, layer.dh, layer.dw, layer.sh, layer.sw
    if kh == 1 and kw == 1:
        return 0, 0, 0, 0
    A = act_bits
    hin, win, hout, wout = layer.hin, layer.win, layer.hout, layer.wout
    cf = layer.cin // simd_swu  # channel_factor
    buffer_width = simd_swu * A
    if parallel_window:
        # get_buffer_depth()'s OWN "parallel" formula -- see docstring for why
        # this is NOT bram_estimation()'s separate inline "parallel" calc.
        buffer_depth = ((kh - 1) * dh * win + (kw - 1) * dw) * cf + 2
        swu_cycles = hin * win * cf + 2
    else:
        buffer_min_size = ((kh - 1) * dh * win + (kw - 1) * dw + 1) * cf
        buffer_depth = (
            buffer_min_size
            + max(0, ((sw - 1) - kh * kw) * cf)
            + max(0, ((sh - 1) * win - kh * kw) * cf)
        )
        max_cycles = max(wout * kw * kh * cf, sw * win * cf)
        if depthwise:
            max_cycles += wout * (sw - 1) * (cf - 1)
        swu_cycles = buffer_min_size + hout * max_cycles
        if depthwise:
            swu_cycles += (sh - 1) * win * cf
    if ram_style == "auto_efficient":
        # Deterministic geometry-driven pick (2026-09-17 addition), NOT real
        # FINN's own "auto" (which never considers URAM at all, see the
        # ram_style docstring paragraph above and MVAU's own identical
        # blind spot) -- compute this EXACT buffer's real block/ultra
        # primitive count and take whichever is smaller, same tie-breaking
        # convention as layer_cost_pe_simd_auto_ram's own weight-memory pick
        # (strict "<", ties go to block). This is what makes "does the
        # geometry favor URAM" a real per-layer computation instead of a
        # free ILP variable CBC would otherwise have to rediscover for every
        # single layer despite it being a pure function of buffer shape.
        bram18_if_block = _finn_buffer_bram18(buffer_width, buffer_depth)
        uram18_if_ultra = _finn_buffer_uram18(buffer_width, buffer_depth)
        ram_style = "ultra" if uram18_if_ultra < bram18_if_block else "block"
    ram_luts = buffer_width * math.ceil(buffer_depth / 38) if ram_style == "distributed" else 0
    swu_lut = (300 + ram_luts) * _SWU_LUT_DERATE
    swu_bram18 = _finn_buffer_bram18(buffer_width, buffer_depth) if ram_style in ("block", "auto") else 0
    swu_uram18 = _finn_buffer_uram18(buffer_width, buffer_depth) if ram_style == "ultra" else 0
    return swu_lut, swu_bram18, swu_uram18, int(swu_cycles)


def _thresholding_rtl_cost(pe: int, output_bits: int, ram_style: str = "block") -> tuple[float, float, float]:
    """(lut, bram18, uram18) of the STANDALONE Thresholding_rtl node that
    follows this layer's MVAU/VVAU under noActivation=1 -- empirical, a real
    function of BOTH pe and numSteps=2**output_bits-1 (2026-09-17 refit, see
    _THR_RTL_LUT_BASE_PER_PE/_THR_RTL_LUT_PER_NUMSTEP_PE/_THR_RTL_BRAM18_
    PER_PE_NUMSTEP's own module-level comment for the real-data basis, why
    the OLD pe-only/flat-BRAM version was confirmed wrong not just
    imprecise, and why FINN's own LUTRAM-count estimator is not used
    instead).

    ram_style (2026-09-17/18 additions, additive/opt-in, default "block"
    preserves prior behavior exactly): "block" returns the real, validated
    bram18 formula with uram18=0, matching every real node checked (FINN's
    own per-stage min-waste primitive selection never picked URAM here,
    depth_trigger_uram=0 throughout). "ultra" is KEPT for provenance but is
    DEAD -- real Vivado synthesis of a Thresholding_rtl node with
    ram_style="ultra" FAILS outright (URAM cannot be ROM; see finn_milp.py's
    RAM_STYLES comment) -- no live caller should ever pass it anymore.
    "distributed" (2026-09-18) returns bram18=uram18=0 and adds a DERIVED
    (not fit) LUTRAM-primitive-count term directly into `lut` instead --
    see _THR_RTL_LUTRAM_PER_PE_NUMSTEP's own comment for the derivation and
    why, unlike "ultra", this one is a real, synthesis-safe option (just
    never yet exercised in real hardware, so still uncalibrated)."""
    num_steps = 2 ** output_bits - 1
    lut = pe * (_THR_RTL_LUT_BASE_PER_PE + _THR_RTL_LUT_PER_NUMSTEP_PE * num_steps)
    if ram_style == "ultra":
        return lut, 0.0, _THR_RTL_URAM_PER_PE_NUMSTEP * pe * num_steps
    if ram_style == "distributed":
        return lut + _THR_RTL_LUTRAM_PER_PE_NUMSTEP * pe * num_steps, 0.0, 0.0
    bram18 = _THR_RTL_BRAM18_PER_PE_NUMSTEP * pe * num_steps
    return lut, bram18, 0.0


def conv_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
    force_dsp: bool = False, impl_style: ImplStyle = IMPL_STYLE_RTL, act_signed: bool = False,
    no_activation: bool = True, ram_style_thresholds: str = "auto",
    swu_ram_style: str = "distributed", thr_ram_style: str = "block",
) -> dict:
    """The general per-layer cost, given EXPLICIT PE/SIMD (the actual
    folding decision variables a folding search chooses over) instead of
    just the two folding-preset endpoints. conv_cost/_pq_for_folding above
    are now thin wrappers around this for the two presets used elsewhere
    (per-stage HAWQ bit-width search); a folding ILP wants the full
    (PE, SIMD) space, not just those two points.

    cycles ~= ceil(H_out*W_out/M) * ceil(C_out/PE) * ceil(Q_max/SIMD) --
    FINN's own analytical per-layer cycle estimate (same category as its
    real estimate_layer_cycles.json report): each of the H_out*W_out output
    pixels needs one pass per PE-group of output channels times one pass
    per SIMD-group of the reduction. Using ceil() rather than requiring
    PE/SIMD to be exact divisors keeps this usable for arbitrary values,
    though every caller in this codebase only ever passes divisors (see
    `divisors()` above), where ceil reduces to exact division anyway.

    ram_style picks which FPGA memory holds the weight tile (wm_bram18 vs
    wm_uram18), mirroring real FINN's mutually-exclusive bram_estimation()/
    uram_estimation() (see matrixvectoractivation.py): "block" (default,
    unchanged behavior) puts weights in BRAM, wm_uram18=0; "ultra" puts them
    in URAM instead (wm_uram18 = ceil(mem_width/72) * ceil(omega/4096), the
    exact formula real FINN's uram_estimation() uses), wm_bram18=0. LUT cost
    is IDENTICAL either way (confirmed via direct FINN source read: real
    FINN's lut_estimation() only adds an extra term for ram_style=
    "distributed", not "ultra") -- URAM is a free swap in this cost model
    other than needing its own separate resource budget.

    swu_ram_style (2026-09-17 addition, additive/opt-in, default
    "distributed" preserves exact prior behavior) is a SEPARATE choice for
    the SWU's own line-buffer memory (swu_bram18/swu_uram18) -- a different
    physical memory on this same layer from the weight tile `ram_style`
    controls above. Deliberately NOT tied to `ram_style`: see _finn_swu's
    own docstring for why real calibration data (43/43 real SWU nodes at
    ram_style="distributed" regardless of the paired MVAU's own ram_style)
    rules out coupling them. finn_milp.py's ILP passes "auto_efficient"
    here for every layer (a deterministic, geometry-driven block-vs-ultra
    pick, not a free ILP variable -- see _finn_swu's own docstring for
    exactly what that means) rather than a fixed style.

    thr_ram_style (2026-09-17 addition, additive/opt-in, default "block"
    preserves exact prior behavior) is a THIRD, independent memory choice --
    for the standalone Thresholding_rtl node's own memory (thr_bram18/
    thr_uram18), a different physical memory again from both the weight
    tile and the SWU line buffer. UNLIKE swu_ram_style, this one IS a real,
    free per-layer ILP choice in finn_milp.py (not a deterministic
    "auto_efficient" pick) -- see _thresholding_rtl_cost's own docstring for
    why: FINN's own local per-primitive min-waste selection never favors
    URAM here, but that is the wrong criterion for a design where BRAM is
    the globally scarce resource and URAM sits mostly idle -- a genuine
    resource-tradeoff decision the ILP's own joint optimization is better
    positioned to make than any local per-node heuristic.

    force_dsp mirrors real FINN's resType="dsp" nodeattr (see
    matrixvectoractivation_hls.py's lut_estimation()): zeroes mvu_lut's
    LUT-based multiplier term (mult_luts) since the multiplication moves to
    DSP48 slices instead -- pass the SAME force_dsp used elsewhere for this
    build (e.g. the one already threaded into calibrated_lut/
    calibrated_bram18k) for consistency; the two used to disagree (raw
    mvu_lut always included a multiplier term regardless of force_dsp,
    only the separate calibration factor knew about it).

    impl_style (default RTL, the go-forward build target) only changes the
    DSP count (FINN's MVAU_rtl.lut_estimation() is 0 -- there is no RTL LUT
    model, the HLS one is used as the proxy for both) and implies DSP
    multipliers. Depthwise layers are forced to HLS (VVAU_rtl is
    Versal-only). act_signed feeds FINN's accumulator-width bound (this
    repo's post-ReLU activations are UINT).

    Every conv layer defaults to noActivation=1 (no_activation=True), i.e. a
    separate Thresholding_rtl node (PE assumed == this layer's PE) priced
    empirically as thr_lut/thr_bram18/thr_uram18 and folded into
    total_lut. cycles = max(MVAU cycles, SWU cycles): a layer runs at the
    speed of its slowest node.

    no_activation=False (2026-09-17 addition, additive/opt-in -- every
    existing caller keeps no_activation=True by default, i.e. byte-identical
    behavior) models FINN's OTHER real regime: the activation/threshold is
    FUSED into this MVAU/VVAU node itself (noActivation=0 in real FINN's own
    nodeattr) instead of living in a separate standalone node. This is only
    a legal FINN configuration under the HLS backend (impl_style="hls") --
    MVAU_rtl/VVAU_rtl structurally require noActivation=1, see the module
    docstring and finn_milp.py's own eligibility filter for the ILP's
    "rtl_dsp_noact1" vs "hls_lut_noact0" variants; this function itself does
    NOT enforce that (it's a pure cost formula, calling it with an illegal
    impl_style/no_activation combination just produces a number nothing in
    real FINN could build).

    When no_activation=False: thr_lut/thr_bram18/thr_uram18 are all 0 (no
    separate node exists in the graph any more -- its resources are gone,
    not moved), and thr_pe is 0 (meaningless, no such node). Instead, real
    FINN's OWN MVAU_hls.lut_estimation() (transcribed directly, container
    source read) adds a "thr_luts + comp_luts" term INSIDE this node's own
    mvu_lut, gated on ram_style_thresholds (FINN's own nodeattr for where
    the fused threshold memory lives, default "auto"): thr_luts/comp_luts
    are literally 0 unless ram_style_thresholds=="distributed" -- FINN's OWN
    static resource estimator has NO modeled LUT/BRAM cost for fused
    thresholds under its own default ("auto") or "block" styles (Vivado
    decides at synthesis time; FINN's own analytical model just doesn't
    price it). This is a real, confirmed blind spot in FINN's own estimator,
    not a gap specific to this repo's calibration -- so at the default
    ram_style_thresholds="auto", no_activation=False is modeled as PURE
    savings (the standalone node's cost disappears, nothing is added in its
    place) exactly mirroring what FINN's own report would show. Only pass
    ram_style_thresholds="distributed" to exercise the real fused-LUT term;
    B (the output activation bit-width FINN's own formula uses,
    self.get_output_datatype().bitwidth()) is approximated here as this
    layer's own `act_bits` -- this cost model has no separate output-
    precision axis distinct from a layer's own (weight_bits, act_bits) pair
    (see finn_milp.py's module docstring on act_bits' own INPUT-stream
    convention) -- a real approximation, not a FINN-source transcription,
    and untested against hardware; treat it as provisional if ever used."""
    W, A = weight_bits, act_bits
    P, Q = pe, simd
    M = 1  # spatial replication -- already minimal in every convention this file implements
    depthwise = layer.groups > 1
    if depthwise:
        impl_style = IMPL_STYLE_HLS  # _vvu_rtl_possible(): VVAU_rtl needs a Versal DSP58 -- always VVAU_hls on xczu7ev

    # SWU (ConvolutionInputGenerator_rtl). Its SIMD is a DIFFERENT axis from
    # the MVAU's: it must divide IFMChannels (cin), while MVAU SIMD ranges over
    # cin*kh*kw. Dense conv: MVAU SIMD up to cin -> SWU SIMD = the part of it
    # that divides cin (gcd; a non-divisor forces the SWU narrower and a DWC
    # in between); MVAU SIMD beyond cin (spanning kernel elements) is only
    # feedable with the SWU in parallel_window mode (whole window per cycle,
    # SIMD == cin, select_impl_style() asserts exactly that). Depthwise: the
    # SWU emits PE channels per cycle (== VVAU PE, see swu_max_simd_depthwise);
    # VVAU SIMD > 1 (several kernel elements per cycle) again needs
    # parallel_window.
    if depthwise:
        simd_swu = P
        parallel_window = Q > 1
    else:
        parallel_window = Q > layer.cin
        simd_swu = layer.cin if parallel_window else math.gcd(Q, layer.cin)
    swu_lut, swu_bram18, swu_uram18, swu_cycles = _finn_swu(
        layer, A, simd_swu, depthwise, parallel_window, ram_style=swu_ram_style,
    )

    # Weight memory (mem_mode internal_decoupled): FINN's base-class
    # bram_estimation()/uram_estimation(), identical for the HLS and RTL
    # backends. omega = words, mem_width = bits per word.
    omega = (layer.kh * layer.kw * (layer.cin // layer.groups) * layer.cout) / (Q * P)
    mem_width = Q * W * P
    if ram_style == RAM_STYLE_ULTRA:
        wm_bram18 = 0
        wm_uram18 = math.ceil(mem_width / 72) * math.ceil(omega / 4096)
    else:
        wm_bram18 = _finn_wm_bram18(omega, mem_width, depthwise)
        wm_uram18 = 0

    # mvu_lut: real FINN's MatrixVectorActivation_hls/VectorVectorActivation_
    # hls.lut_estimation() (Blott et al. FINN-R, confirmed via container
    # source read 2026-09-16) -- c0 + c1*P*(mult_luts + addertree_luts +
    # acc_luts) -- NOT the old flat "300 + 1.1*P*Q*W*A" stand-in. mult_luts
    # is the LUT-based multiplier array, ZERO when force_dsp (multiplication
    # moves into DSP48 slices instead); addertree_luts is the SIMD-wide
    # reduction tree; acc_luts is the accumulator register width, which
    # grows with log2(MW) where MW is the FULL reduction depth
    # (max_simd(layer) -- the whole cin/groups*kh*kw, NOT the folded Q) --
    # a real dependence on kernel/channel geometry the old flat formula had
    # no way to express (root cause of the dense-vs-separable LUT
    # under-prediction, see repo memory finn_calibrated_8way_build_status.md's
    # "LUT model mismatch: CONFIRMED root cause" section). thr_luts/
    # comp_luts (FUSED-threshold LUTs, (2^B-1)*acc_bits per PE) are 0.0 here
    # whenever no_activation=True (the module default, unchanged) or
    # ram_style_thresholds != "distributed" (also the default -- see
    # conv_cost_pe_simd's own docstring for why that's FINN-faithful, not
    # just unmodeled) -- see thr_luts_fused/comp_luts_fused below for the
    # no_activation=False + "distributed" case. NOTE the
    # pre-2026-09-17 production builds did NOT satisfy that: 172/173 nodes in
    # hardware/mvau_lut_calibration_dataset*.csv have outputDataType=UINTx,
    # i.e. fused thresholds -> forced MVAU_hls (RTL needs noActivation=1) and
    # a PE*2^B-scaled LUT blow-up that the former "imbalance_luts" term was
    # chasing empirically. Standalone-threshold builds are priced via
    # thr_lut below instead.
    mw = max_simd(layer)  # full reduction depth (cin/groups * kh * kw) -- same domain as max_simd(), NOT the folded Q
    use_dsp = force_dsp or impl_style == IMPL_STYLE_RTL  # the RTL MVU is DSP-only
    mult_luts = 0 if use_dsp else Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
    addertree_luts = (W + A) * (2 * Q - 1)
    # FINN: alpha = log2(MW) + W + A - 1 - int(idt.signed()), acc_bits =
    # min(accDataType width, ceil(alpha + log2(1+2^-alpha) + 1)) -- the
    # https://arxiv.org/abs/2301.13376 bound, capped at the INT32 default.
    # This repo's post-ReLU activations are UNSIGNED (165/173 calibration
    # nodes UINT4/6/8), hence act_signed=False by default.
    alpha = math.log2(mw) + W + A - 1 - int(act_signed)
    acc_bits = min(32, math.ceil(alpha + math.log2(1 + 2 ** -alpha) + 1))
    acc_luts = acc_bits

    # thr_luts_fused/comp_luts_fused: real FINN's MVAU_hls.lut_estimation()
    # own fused-threshold term (transcribed from matrixvectoractivation_hls.py,
    # container source read 2026-09-17), additive/opt-in via no_activation --
    # see conv_cost_pe_simd's own docstring. 0.0 unless no_activation=False
    # AND ram_style_thresholds=="distributed" (FINN's own condition, not this
    # repo's invention: its lut_estimation() only counts this term for
    # "distributed", leaving fused thresholds under the "auto"/"block"
    # default un-costed). B approximates FINN's own output-datatype bitwidth
    # with this layer's own act_bits (see docstring caveat).
    if no_activation or ram_style_thresholds != "distributed":
        thr_luts_fused, comp_luts_fused = 0.0, 0.0
    else:
        tmem = layer.cout // P
        B = A
        thr_luts_fused = (2 ** B - 1) * acc_bits * math.ceil(tmem / 64)
        comp_luts_fused = (2 ** B - 1) * acc_bits

    # (The former empirical "imbalance_luts" term -- (k_pe*PE/SIMD +
    # k_mh*max(0,MH-mw))*A, pooled R^2=0.779 on the two fused-threshold
    # calibration sets -- was REMOVED 2026-09-17. It was a proxy for FINN's
    # own thr_luts/comp_luts: LUT/PE in that data scales ~3x per +2 output
    # bits, every "SIMD=1 cliff" row had 8-bit outputs, and geometry-
    # identical MVAU_rtl/noActivation probes show none of it. See
    # analysis/hardware_calibration/ for the exploration.)
    # FINN-R constants (MVAU_hls/VVAU_hls.lut_estimation()). c2, FINN's extra
    # LUTRAM term, only applies to ram_style="distributed" (or embedded
    # weights <= 128 words) -- neither is selectable here (block/ultra), so 0.
    c0, c1 = 300, 1.1
    mvu_lut = c0 + c1 * M * P * (mult_luts + addertree_luts + acc_luts + thr_luts_fused + comp_luts_fused)

    # RTL LUT derating (2026-09-17, REFIT 2026-09-18): MVAU_rtl.lut_estimation()
    # is literally `return 0` in FINN v1.0.0-alpha -- there is NO real RTL LUT
    # model, so the HLS formula above is used as a stand-in for the RTL
    # backend too. real_LUT (the MVAU_rtl node ALONE, per its own Vivado
    # hierarchy row -- NOT swu_lut/thr_lut, which are genuinely separate FINN
    # graph nodes) comes in consistently LOWER than this formula, across
    # every node regardless of MH/PE. Makes physical sense: MVAU_rtl's
    # add_multi.sv folds accumulation into the DSP58 cascade chain, doing in
    # hardware what addertree_luts/acc_luts assume needs separate LUT logic.
    #
    # First fit (2026-09-17, briefly): n=7, one probe (hardware/mvau_lut_
    # calibration_dataset_s12_context_dense_int6_pemh_simd1.csv), one
    # architecture, one folding pattern (PE=MH/SIMD=1, W6A6) -- OLS-through-
    # origin gave 0.581, R^2=0.747, and was explicitly flagged PROVISIONAL
    # pending the full production rebuild.
    # REFIT (2026-09-18): that production rebuild now exists (hardware/
    # datasets/mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_
    # alpha025_rtl_mvau_noact1_extended.csv, 88 real MVAU_rtl nodes, many
    # (PE, SIMD, weight_bits, act_bits, MW) combinations from the real 8-way
    # deployed build) -- refitting on it (same OLS-through-origin convention)
    # gives 0.4868, R^2=0.887 (up from 0.747 when the OLD 0.581 constant is
    # scored against this same larger dataset) -- a tighter fit on far more
    # data, not a different formula. At the aggregate level: this build's 88
    # real MVAU_rtl nodes sum to 59,630 real LUT; the OLD constant (0.581)
    # predicted 71,072 (+19%), the REFIT constant (0.4868) predicts 59,552
    # (+0.1%). Only applies to impl_style="rtl" -- VVAU_hls (depthwise) stays
    # unscaled (no real depthwise MVAU_rtl data exists yet either way, since
    # VVAU_rtl needs Versal/DSP58 and this repo targets xczu7ev); the
    # non-depthwise impl_style="hls"+use_dsp=False (LUT-mult) case gets its
    # OWN separate derate below instead, since it's a structurally different
    # regime (MVAU_hls.lut_estimation() is a REAL formula, unlike MVAU_rtl's
    # `return 0`, so there was never a reason to assume the same bias applies).
    _RTL_MVU_LUT_DERATE = 0.4868
    if impl_style == IMPL_STYLE_RTL:
        mvu_lut *= _RTL_MVU_LUT_DERATE

    # HLS LUT-mult derating (2026-09-18): MVAU_hls.lut_estimation() (this
    # formula) is a REAL FINN model, unlike MVAU_rtl's `return 0` stand-in
    # above -- but it's still an analytical estimate, and real synthesis
    # diverges from it too. First real per-node ground truth (hardware/
    # datasets/mvau_variant_matrix_dataset_extended.csv, the noact1_hls_lut_*
    # combos: standalone Thresholding_rtl + MVAU_hls forced resType=lut,
    # 3 real conv layers (MH/MW = 8/32, 8/72, 32/8) x 3 fold strategies
    # (pemh_simd1/balanced/pe1_simdmw) = 9 real MVAU_hls nodes, W8A8):
    # real_LUT comes in at 0.83-1.48x of this formula's raw prediction
    # (OLS-through-origin fit 0.7402, R^2=0.885) -- consistently LOWER,
    # same direction/rough magnitude as the RTL derate above, though this is
    # a genuinely separate fit (different node type, different formula
    # branch: mult_luts is nonzero here, 0 for RTL). PROVISIONAL: n=9, one
    # probe network, one bit-width (W8A8), PE/SIMD in {1,8,32,72} only --
    # refit if/when a production hls_lut_noact0 build exists. Only applies
    # when use_dsp=False (the actual LUT-mult case) -- impl_style="hls" with
    # use_dsp=True (the VARIANT_HLS_DSP_NOACT0 placeholder, never live in the
    # ILP) stays unscaled, no real motivating data checked for it yet.
    _HLS_MVU_LUT_MULT_DERATE = 0.7402
    if impl_style == IMPL_STYLE_HLS and not use_dsp:
        mvu_lut *= _HLS_MVU_LUT_MULT_DERATE

    # Standalone Thresholding_rtl that consumes this layer's accumulators
    # under noActivation=1. Its PE is NOT a free choice and NOT simply == the
    # MVAU's PE: the MVAU emits one fold of P outputs every SF = MW/Q cycles,
    # so per output pixel the threshold node has NF*SF = (MH/P)*(MW/Q) cycles
    # to process MH channels at PE_thr per cycle -> it keeps up iff
    # PE_thr >= P*Q/MW (and FINN requires PE_thr | NumChannels). The smallest
    # such divisor is what SetFolding-style balancing would pick; anything
    # larger only costs LUT. NOTE for the build side: FINN's default
    # Thresholding PE is 1, and the folding configs carry no Thresholding
    # entries -- the bridge must write thr_pe (returned below) explicitly, or
    # a PE=1 node throttles any layer whose MVAU needs PE_thr > 1.
    #
    # no_activation=False: no standalone Thresholding node exists in the
    # graph at all (the activation is fused into this MVAU/VVAU node
    # instead, see thr_luts_fused/comp_luts_fused above) -- its resources
    # are simply gone, not moved, so thr_lut/thr_bram18/thr_uram18=0 and
    # thr_pe=0 (meaningless, no such node to give a PE to).
    if no_activation:
        thr_pe = next(d for d in divisors(layer.cout) if d * mw >= P * Q)
        # output_bits=A (this layer's own act_bits): see _thresholding_rtl_cost's
        # own docstring for why this is the best available proxy, not an exact
        # match, for the real outputDataType bitwidth the standalone node stores.
        thr_lut, thr_bram18, thr_uram18 = _thresholding_rtl_cost(thr_pe, A, ram_style=thr_ram_style)
    else:
        thr_pe, thr_lut, thr_bram18, thr_uram18 = 0, 0.0, 0.0, 0
    total_lut = swu_lut + mvu_lut + thr_lut

    # DSP. HLS: MVAU_hls.dsp_estimation() P*Q*ceil((W+A)/48) (one DSP48E2 per
    # MAC lane; VVAU_hls's own formula is P*ceil((W+A)/48), no SIMD factor --
    # transcribed as-is), zero unless the multiply is on DSP.
    #
    # RTL: FINN's OWN MVAU_rtl.dsp_estimation() source (util.basic.
    # get_dsp_block(fpgapart)=="DSP58" -> P*ceil(Q/3); else (DSP48E1/E2,
    # which is what xczu7ev-ffvc1156-2-e resolves to) -> ceil(P/4)*Q, NO
    # bit-width dependence at all) says 4 PE lanes per DSP48E2, unconditional
    # on W/A. Real per-node ground truth (hardware/probes/mvau_lut_
    # calibration_dataset_s12_context_dense_int6_pemh_simd1.csv, 7 real
    # MVAU_rtl nodes, W6A6) shows real_DSP = ceil(P/2)*Q EXACTLY on every
    # node (16/4/4/16/4/4/16) -- exactly 2x FINN's own formula, at every
    # (P,Q) pair tested. A PREVIOUS version of this code branched on bit-
    # width (4 lanes for <=4-bit, 2 for >4-bit) based on the 8-bit pemh
    # aggregate probes also landing at 2x -- that branch had NO real
    # evidence behind its <=4-bit half: every <=4-bit data point available
    # (dense_int4/separable_int4, PE=1 throughout) can't distinguish
    # ceil(1/4) from ceil(1/2) (both round up to 1), so "4 lanes at <=4-bit"
    # was an untested assumption, not a finding. Corrected 2026-09-17 to a
    # flat, unconditional 2 lanes/DSP -- the only value with real per-node
    # support at high PE, and consistent (if uninformative) at PE=1 too.
    # Refit if a real high-PE low-bit (e.g. int4_pemh) probe ever exists.
    if impl_style == IMPL_STYLE_RTL:
        mvu_dsp = M * math.ceil(P / 2) * Q
    elif use_dsp:
        mvu_dsp = M * P * math.ceil((W + A) / 48) if depthwise else M * P * Q * math.ceil((W + A) / 48)
    else:
        mvu_dsp = 0

    total_pe = P * M
    total_simd_lanes = P * Q * M
    mvu_cycles = math.ceil(layer.hout * layer.wout / M) * math.ceil(max_pe(layer) / P) * math.ceil(max_simd(layer) / Q)
    cycles = max(mvu_cycles, swu_cycles)
    return {
        "total_pe": total_pe, "total_simd_lanes": total_simd_lanes,
        "swu_bram18": swu_bram18, "swu_uram18": swu_uram18, "wm_bram18": wm_bram18, "wm_uram18": wm_uram18,
        "thr_bram18": thr_bram18, "thr_uram18": thr_uram18,
        "swu_lut": swu_lut, "mvu_lut": mvu_lut, "thr_lut": thr_lut, "mp_lut": 0,
        "total_lut": total_lut, "mvu_dsp": mvu_dsp, "total_dsp": mvu_dsp,
        "cycles": cycles, "mvu_cycles": mvu_cycles, "swu_cycles": swu_cycles,
        "impl_style": impl_style, "simd_swu": simd_swu, "thr_pe": thr_pe, "acc_bits": acc_bits,
    }


def conv_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED, force_dsp: bool = False,
) -> dict:
    """Conv2d cost (Eq. 1/4/5 of finn_cost_formulae.md's source paper) at
    one of the two folding PRESETS (unfolded/serial) -- see
    conv_cost_pe_simd for the general (arbitrary PE, SIMD) version a
    folding search needs. ConvTranspose2d is handled by the caller
    pre-converting its geometry into the equivalent zero-inserted dense
    conv (see conv_transpose_cost) before calling this.

    Uses the GENERAL BRAM_wm formula (omega = K^2*C*C'/(Q*P)), not a fixed
    omega=1 shortcut -- at FOLDING_UNFOLDED this still simplifies to
    omega=1 automatically (Q*P always equals the full weight volume there),
    so FOLDING_UNFOLDED's numbers are unchanged/still verified; at
    FOLDING_SERIAL (Q=P=1) omega is the FULL weight volume, correctly
    reflecting "the whole layer's weights get streamed through one PE over
    many cycles" instead of loaded all at once."""
    Q, P = _pq_for_folding(layer, folding)
    return conv_cost_pe_simd(layer, weight_bits, act_bits, P, Q, force_dsp=force_dsp)


def conv_transpose_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED, force_dsp: bool = False,
) -> dict:
    """ConvTranspose2d modeled as zero-insertion + ordinary stride-1 conv
    (Dumoulin & Visin) -- see finn_cost_formulae.md's own derivation. Only
    K=S, p=0 transposed convs are used anywhere in this architecture (up4.
    up.0, up5.up.0, final), matching that file's own confirmed case."""
    assert layer.kh == layer.sh and layer.kw == layer.sw, (
        f"{layer.name}: conv_transpose_cost only implements the K=S,p=0 case this "
        f"architecture actually uses (got kh={layer.kh},sh={layer.sh})."
    )
    n_eff_h = (layer.hin - 1) * layer.sh + 1 + 2 * (layer.kh - 1)
    n_eff_w = (layer.win - 1) * layer.sw + 1 + 2 * (layer.kw - 1)
    equivalent = LayerGeometry(
        op_type="Conv2d", name=layer.name, stage=layer.stage,
        cin=layer.cin, hin=n_eff_h, win=n_eff_w, cout=layer.cout, hout=layer.hout, wout=layer.wout,
        kh=layer.kh, kw=layer.kw, sh=1, sw=1, dh=1, dw=1,
    )
    return conv_cost(equivalent, weight_bits, act_bits, folding, force_dsp=force_dsp)


def maxpool_cost(layer: LayerGeometry, act_bits: int) -> dict:
    """MaxPool2d: SWU + comparator array, no MVAU/weights -- act_bits only
    (no weight_bits, there's nothing to quantize). No P/Q/folding at all --
    pooling was never a folded MVAU to begin with, identical in every
    convention. cycles ~= H_out*W_out (one comparison pass per pixel, M=1)."""
    A = act_bits
    M = 1
    k_eff = _k_eff(layer.kh, layer.dh)
    swu_bram18 = M * (math.ceil(k_eff / layer.sh) + 1) * math.ceil(layer.sh * layer.win / 512) * math.ceil(layer.cin * A / 36)
    swu_lut = M * 426
    mp_lut = M * A * layer.cin
    total_lut = swu_lut + mp_lut
    cycles = math.ceil(layer.hout * layer.wout / M)
    return {
        "total_pe": 0, "total_simd_lanes": 0,
        "swu_bram18": swu_bram18, "wm_bram18": 0, "wm_uram18": 0,
        "swu_lut": swu_lut, "mvu_lut": 0, "mp_lut": mp_lut,
        "total_lut": total_lut, "mvu_dsp": 0, "total_dsp": 0,
        "cycles": cycles,
    }


def layer_cost(
    layer: LayerGeometry, weight_bits: int, act_bits: int, folding: Folding = FOLDING_UNFOLDED, force_dsp: bool = False,
) -> dict:
    if layer.op_type == "Conv2d":
        return conv_cost(layer, weight_bits, act_bits, folding, force_dsp=force_dsp)
    if layer.op_type == "ConvTranspose2d":
        return conv_transpose_cost(layer, weight_bits, act_bits, folding, force_dsp=force_dsp)
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")


def layer_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
    force_dsp: bool = False, **kw,
) -> dict:
    """Like layer_cost, but for an explicit (PE, SIMD) folding choice
    (what a real folding search sweeps over) instead of one of the two
    presets. MaxPool2d has no PE/SIMD/weights at all -- pe/simd/ram_style
    are ignored for it (asserted to be the sentinel max_pe/max_simd=1 by
    the caller's own candidate enumeration, see folding_ilp.py's PoolCost,
    so this never silently drops a real folding choice). **kw is passed
    straight to conv_cost_pe_simd (impl_style/act_signed/no_activation/
    ram_style_thresholds/swu_ram_style/thr_ram_style)."""
    if layer.op_type == "Conv2d":
        return conv_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style, force_dsp=force_dsp, **kw)
    if layer.op_type == "ConvTranspose2d":
        n_eff_h = (layer.hin - 1) * layer.sh + 1 + 2 * (layer.kh - 1)
        n_eff_w = (layer.win - 1) * layer.sw + 1 + 2 * (layer.kw - 1)
        equivalent = LayerGeometry(
            op_type="Conv2d", name=layer.name, stage=layer.stage,
            cin=layer.cin, hin=n_eff_h, win=n_eff_w, cout=layer.cout, hout=layer.hout, wout=layer.wout,
            kh=layer.kh, kw=layer.kw, sh=1, sw=1, dh=1, dw=1,
        )
        return conv_cost_pe_simd(equivalent, weight_bits, act_bits, pe, simd, ram_style, force_dsp=force_dsp, **kw)
    if layer.op_type == "MaxPool2d":
        return maxpool_cost(layer, act_bits)
    raise ValueError(f"Unknown op_type {layer.op_type!r} for layer {layer.name}")


def layer_cost_pe_simd_auto_ram(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, force_dsp: bool = False, **kw,
) -> dict:
    """Like layer_cost_pe_simd, but picks ram_style per-layer instead of
    taking it as a fixed input -- the "leave memory type as auto" default
    for cost-model estimates going forward (standing convention, 2026-09-15).

    LUT is IDENTICAL between "block" and "ultra" (real FINN's lut_estimation()
    only adds an extra term for ram_style="distributed", not "ultra" -- see
    conv_cost_pe_simd's own docstring, confirmed via direct FINN source
    read) -- so "auto" only ever changes which pool (BRAM_18K vs URAM) a
    layer's weight memory lands in, never total_lut/cycles. Picks whichever
    of wm_bram18/wm_uram18 is the SMALLER block count for this layer (a
    stand-in for real FINN's own auto ram_style heuristic, which isn't
    directly accessible from this repo). MaxPool2d has no weight memory at
    all -- ram_style is irrelevant there, both calls would return identical
    results anyway, so it's skipped."""
    block = layer_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style=RAM_STYLE_BLOCK, force_dsp=force_dsp, **kw)
    if layer.op_type == "MaxPool2d":
        return block
    ultra = layer_cost_pe_simd(layer, weight_bits, act_bits, pe, simd, ram_style=RAM_STYLE_ULTRA, force_dsp=force_dsp, **kw)
    chosen = ultra if ultra["wm_uram18"] < block["wm_bram18"] else block
    return {**chosen, "ram_style_chosen": "ultra" if chosen is ultra else "block"}
