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
that follows every MVAU/VVAU under noActivation=1 (empirical per-node
cost from 66 real nodes -- FINN's own LUTRAM-count estimator contradicts
the Vivado reports, see _THR_RTL_LUT_PER_PE). Still not
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


def calibrated_lut(raw_total_lut: float, weight_bits: float, act_bits: float, force_dsp: bool = False) -> float:
    """This model's own raw total_lut, corrected by a derating factor
    interpolated between the avg_bits=3.52/avg_bits=8 real-synthesis
    anchors at avg_bits=(weight_bits+act_bits)/2 -- see the module-level
    comment above for the calibration this is based on and its real scope
    limits.

    force_dsp=True switches to the FORCED-DSP regime's own flat factor
    (see _FORCED_DSP_LUT_FACTOR below) instead -- default False preserves
    this function's exact prior behavior for every existing caller."""
    if force_dsp:
        return raw_total_lut * _FORCED_DSP_LUT_FACTOR
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

# Standalone Thresholding_rtl, EMPIRICAL per-node cost. FINN's own
# Thresholding_rtl.lut_estimation() counts LUTRAM primitives for the
# per-output-bit threshold memories (get_pe_mem_geometries + min-waste
# primitive choice) -- but every real standalone Thresholding_rtl node
# checked shows LUTRAM = 0: Vivado puts the memories in RAMB18/RAMB36 +
# FFs instead and leaves a small amount of logic+SRL per node. Refit
# 2026-09-17 on 79 real nodes (hardware/mvau_swu_threshold_calibration_
# dataset.csv: the S12-separable alpha025 production build's partitions
# 0/2-7, PLUS the s12_context_dense_int6_pemh probe's own 13 threshold
# nodes -- up from the original 66-node, single-build sample). ALL 79 are
# still PE=1 (FINN's folding-config default when nothing overrides
# Thresholding's own PE -- see conv_cost_pe_simd's thr_pe derivation and
# its own note that the bridge must write thr_pe explicitly into the
# build), so per-PE scaling is still UNVALIDATED, carried over unchanged
# from the original single-build fit as a structural assumption (the RTL
# core replicates its comparator/memory per PE lane) rather than
# re-derived here. LUT barely moved (mean 69.3 vs the old 68.0 -- the
# original fit already generalized well to 5x the data, across a
# genuinely different build). BRAM moved more (mean 3.68 vs the old
# 141/66=2.14) -- the original 66-node sample undercounted it.
_THR_RTL_LUT_PER_PE = 69.3165
_THR_RTL_BRAM18_PER_NODE = 3.6835


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


_SWU_LUT_DERATE = 0.755  # provisional, OLS-through-origin fit, n=43 real ConvolutionInputGenerator_rtl
                          # nodes (hardware/mvau_swu_threshold_calibration_dataset.csv) -- real Vivado
                          # LUT comes in 51-104% of this formula's estimate (mean 70%), same direction
                          # as _RTL_MVU_LUT_DERATE (real RTL synthesis under FINN's own analytical
                          # estimate). Refit once the full production rebuild lands.


def _finn_swu(layer: LayerGeometry, act_bits: int, simd_swu: int, depthwise: bool, parallel_window: bool) -> tuple[int, int, int]:
    """FINN ConvolutionInputGenerator_rtl (the sliding-window unit feeding
    this layer's MVAU/VVAU): (swu_lut, swu_bram18, swu_cycles).

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

    ram_style: FINN's ConvolutionInputGenerator_rtl.lut_estimation() adds a
    LUTRAM term (buffer_width*ceil(buffer_depth/38)) ONLY for ram_style=
    "distributed", and its bram_estimation() returns 0 for that same style
    (BRAM otherwise, mutually exclusive -- same shape as the MVAU's own
    block-vs-ultra split, just LUT-vs-BRAM here). This function ALWAYS
    takes the distributed/LUT branch and returns swu_bram18=0: every real
    SWU node checked (43/43) used ram_style="distributed" with
    real_BRAM18==0 -- FINN appears to default the SWU to this style with
    nothing in this repo's build flow overriding it, and there is no
    ram_style parameter threaded in here to select the block/BRAM
    alternative (only the MVAU's own weight-memory ram_style is a real ILP
    decision variable)."""
    kh, kw, dh, dw, sh, sw = layer.kh, layer.kw, layer.dh, layer.dw, layer.sh, layer.sw
    if kh == 1 and kw == 1:
        return 0, 0, 0
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
    ram_luts = buffer_width * math.ceil(buffer_depth / 38)
    swu_lut = (300 + ram_luts) * _SWU_LUT_DERATE
    return swu_lut, 0, int(swu_cycles)


def _thresholding_rtl_cost(pe: int) -> tuple[float, float, int]:
    """(lut, bram18, uram) of the STANDALONE Thresholding_rtl node that
    follows this layer's MVAU/VVAU under noActivation=1 -- empirical, see
    _THR_RTL_LUT_PER_PE / _THR_RTL_BRAM18_PER_NODE for the real-data basis
    and why FINN's own LUTRAM-count estimator is not used."""
    return _THR_RTL_LUT_PER_PE * pe, _THR_RTL_BRAM18_PER_NODE, 0


def conv_cost_pe_simd(
    layer: LayerGeometry, weight_bits: int, act_bits: int, pe: int, simd: int, ram_style: RamStyle = RAM_STYLE_BLOCK,
    force_dsp: bool = False, impl_style: ImplStyle = IMPL_STYLE_RTL, act_signed: bool = False,
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

    Every conv layer is assumed to have noActivation=1, i.e. a separate
    Thresholding_rtl node (PE assumed == this layer's PE) priced
    empirically as thr_lut/thr_bram18/thr_uram18 and folded into
    total_lut. cycles = max(MVAU cycles, SWU cycles): a layer runs at the
    speed of its slowest node."""
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
    swu_lut, swu_bram18, swu_cycles = _finn_swu(layer, A, simd_swu, depthwise, parallel_window)

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
    # comp_luts (FUSED-threshold LUTs, (2^B-1)*acc_bits per PE) are NOT
    # modeled because noActivation=1 is assumed from 2026-09-17 on. NOTE the
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
    mvu_lut = c0 + c1 * M * P * (mult_luts + addertree_luts + acc_luts)

    # RTL LUT derating (2026-09-17): MVAU_rtl.lut_estimation() is literally
    # `return 0` in FINN v1.0.0-alpha -- there is NO real RTL LUT model, so
    # the HLS formula above is used as a stand-in for the RTL backend too.
    # First real per-node MVAU_rtl ground truth (hardware/
    # mvau_lut_calibration_dataset_s12_context_dense_int6_pemh_simd1.csv, 7
    # nodes, one S12-context probe, W6A6, PE=MH/SIMD=1): real_LUT (the
    # MVAU_rtl node ALONE, per its own Vivado hierarchy row -- NOT swu_lut/
    # thr_lut, which are genuinely separate FINN graph nodes not present in
    # this per-MVAU dataset) comes in at 0.43-0.76x (mean 0.58, OLS-through-
    # origin fit 0.581, R^2=0.747) of this formula -- consistently LOWER
    # across every node regardless of MH/PE, not just the earlier "SIMD=1
    # cliff" nodes. Makes physical sense: MVAU_rtl's add_multi.sv folds
    # accumulation into the DSP58 cascade chain, doing in hardware what
    # addertree_luts/acc_luts assume needs separate LUT logic. PROVISIONAL:
    # n=7, one architecture, one folding pattern (PE=MH/SIMD=1) -- refit
    # once the full production 8-way noActivation/MVAU_rtl rebuild (in
    # progress) gives real per-node data across many more (PE, SIMD, bits)
    # combinations. Does NOT apply to impl_style="hls" (VVAU_hls stays
    # unscaled -- no real depthwise MVAU_rtl data exists yet either way,
    # since VVAU_rtl needs Versal/DSP58 and this repo targets xczu7ev).
    _RTL_MVU_LUT_DERATE = 0.581
    if impl_style == IMPL_STYLE_RTL:
        mvu_lut *= _RTL_MVU_LUT_DERATE

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
    thr_pe = next(d for d in divisors(layer.cout) if d * mw >= P * Q)
    thr_lut, thr_bram18, thr_uram18 = _thresholding_rtl_cost(thr_pe)
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
        "swu_bram18": swu_bram18, "wm_bram18": wm_bram18, "wm_uram18": wm_uram18,
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
    straight to conv_cost_pe_simd (impl_style/act_signed)."""
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
