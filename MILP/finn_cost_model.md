# finn_cost_model.py — rationale, calibration history, and provenance

This file is the companion reference for `finn_cost_model.py`. The `.py` keeps
only the formulas and short comments needed to read them; everything about
*why* a constant has the value it has, what real hardware it was fit against,
and what regimes/branches came and went lives here.

## Overview

Analytical FINN dataflow resource-cost formulas, parameterized by weight
bit-width `W`, activation bit-width `A`, and folding config (`P`, `Q` — how
many output channels / reduction elements are computed in parallel per
cycle), instead of hardcoded 8x8 bits and a single fixed folding.

These are the same formulas already used to produce this repo's real FINN
estimate reports (e.g. `hardware/outputs/quantEnet_original_int8_unfolded_
report/files/enet_finn_fully_unfolded_M1_stage_summary.csv` and its sibling
`finn_cost_formulae.md`, source: Blott et al., "FINN-R: An End-to-End
Deep-Learning Framework for Fast Exploration of Quantized Neural Networks",
ACM TRETS 2018, Sec. 3.2) — verified by hand against that report's own
per-layer CSV (e.g. `down1.reduce.0`: cin=16,cout=16,kh=kw=sh=sw=2,W=A=8 ->
wm_bram18=240, swu_bram18=8, mvu_lut=72390, matches this module's formulas
exactly, at `FOLDING_UNFOLDED`). Reimplemented here as pure functions of
`(W, A, folding)` rather than reading that fixed-8x8/fixed-folding report,
since the HAWQ per-stage search needs the same cost model evaluated at `W,A
in {2,4,8}` — no FINN toolchain/Docker container needed, it's a closed-form
estimate, not an actual FINN build.

Two folding configs, the two ends of the P/Q spectrum:
- `FOLDING_UNFOLDED` (Q=C_in*K_h*K_w, P=C_out): the "fully unfolded" case
  every existing report/estimate in this repo uses — entire reduction and
  all output channels computed in one cycle per output pixel. Maximum
  resource usage, minimum latency.
- `FOLDING_SERIAL` (Q=1, P=1): the most serial case — one reduction element
  and one output channel per cycle. Minimum resource usage, maximum latency.

Both use the same general BRAM_wm formula (`omega = K^2*C*C'/(Q*P)`, not the
"omega=1" shortcut a folding-unaware version would need) — confirmed this
still reduces to the exact verified `FOLDING_UNFOLDED` numbers above (Q*P
always equals the full weight volume there, so omega=1 falls out
automatically, byte-identical to the old hardcoded-omega=1 formula).

**Finding from comparing the two**: BRAM_swu (Eq. 4) does NOT depend on P or
Q at all — only on M, kernel/stride/dilation geometry, and A. So SWU
(line-buffer) BRAM is identical between `FOLDING_UNFOLDED` and
`FOLDING_SERIAL` — folding trades away MVU compute/weight-memory resources,
never the sliding-window buffer. If SWU BRAM alone already exceeds budget,
no amount of folding fixes it — only M (already minimal, =1), A (bit-width),
or the underlying kernel/stride/channel geometry can.

### 2026-09-17 refresh — noActivation regime

Formulas transcribed from the FINN v1.0.0-alpha source checkout at
`<repo>/finn` (`src/finn/custom_op/fpgadataflow/{matrixvectoractivation,
vectorvectoractivation}.py`, `hls/{matrixvectoractivation,
vectorvectoractivation}_hls.py`, `rtl/{matrixvectoractivation,
vectorvectoractivation,convolutioninputgenerator,thresholding}_rtl.py`,
`util/basic.py`). `conv_cost_pe_simd` now models, per conv layer: MVAU/VVAU
LUT (HLS noActivation formula; FINN's `MVAU_rtl.lut_estimation()` is
literally 0, so the HLS formula doubles as the RTL proxy), DSP (HLS:
`P*Q*ceil((W+A)/48)`; RTL DSP48: `ceil(P/lanes)*Q`), weight memory BRAM_18K
(FINN's exact SDP aspect-ratio table) / URAM, the RTL sliding-window unit
(LUT=300, line-buffer BRAM and cycle count as a function of the SWU's own
SIMD), and the standalone `Thresholding_rtl` node that follows every
MVAU/VVAU under `noActivation=1` (empirical per-node cost, a real function
of `pe` AND `numSteps` — FINN's own LUTRAM-count estimator contradicts the
Vivado reports, see the Thresholding section below).

Still not covered: `StreamingDataWidthConverters` (need the successor
layer's folding), FIFOs, shell/interconnect.

The old empirical "imbalance_luts" term is gone: it was measuring
fused-threshold logic inside `MVAU_hls` (`outputDataType=UINTx` on 172/173
calibration nodes), which `noActivation=1` removes at the source — see
`analysis/hardware_calibration/`.

## Auto-resType LUT/BRAM derating (`calibrated_lut` / `calibrated_bram18k`)

Empirical, bit-width-dependent LUT/BRAM_18K derating factors
(real_synthesis / this_model's_own_estimate).

Originally (2026-08-25) a single flat factor calibrated against ONE real
data point (S19 at uniform W8A8, `hardware/results.csv`'s
`s19_double_mid_8way_partitioned_ooc_synth_TOTAL` row, vs. this model
evaluated at the real build's own resolved per-layer PE/SIMD —
`hardware/outputs/s19_8way_partitioned_ooc_20260820_101224/
final_hw_config.json`, PE=SIMD=1 on effectively every MVAU node, i.e.
avg_bits=8 uniform):
- LUT: 830,689 real vs. 100,996 this-model raw -> 8.225x
- BRAM_18K: 906 real vs. 1,495 this-model raw -> 0.606x

A second real data point (same day) proved the flat-factor assumption
wrong: `hardware/results.csv`'s `s19_hawq_block_partition_2_ooc_synth` row
is a real OOC synthesis of partition_id=2 (down2 + stage2.0-2.4 +
stage2.5.reduce.0 — the largest of the 8-way S19 partitions, 23 real MVAU
nodes) built at a real per-block HAWQ bit assignment AND its own real
resolved per-layer folding (`hardware/outputs/
s19_hawq_block_partition_2_ooc_synth_20260824_220316/
hawq_folding_config_partition2.json`, PE=1 but SIMD in {4,6,8,12} — NOT
`FOLDING_SERIAL`).

**Correction (2026-08-25, later same day)**: this anchor was first computed
wrong twice over — (a) assuming that partition's real per-block bits were
uniform W2A2 (they weren't: the real per-block assignment mixed W2/W4
weights and mostly W4A4/some W8 acts across down2/stage2.0-2.5, average
(weight+act)/2 per layer, LUT-weighted across the partition's own 23
layers, is ~3.52, not 2), and (b) evaluating this model at the current
(since-regenerated) `folding_block_s19.json` instead of the real folding
FINN actually built with (`hawq_folding_config_partition2.json`). Both
fixed by recomputing `raw_total_lut`/`raw_total_bram18` directly from this
model's own `layer_cost_pe_simd()`, fed the real per-block bits AND real
per-layer (PE, SIMD) for all 23 layers, `RAM_STYLE_BLOCK` (BRAM_36K=0,
URAM=0 in the real synthesis row, confirming no distributed/ultra RAM was
used):
- avg_bits=3.52 (LUT-weighted mean of (w+a)/2 across the partition's 23
  real layers): LUT 22,436 real vs. 18,352 this-model raw -> 1.223x.
  BRAM_18K 22 real vs. 171 this-model raw -> 0.129x.

The derating factor falls sharply at lower bit-width (LUT ~8.2x at
avg_bits=8 down to ~1.2x at avg_bits=3.52; BRAM ~0.61x down to ~0.13x) —
lines up with the real synthesis notes: FINN's own resType heuristic packs
narrow (2/4-bit) MACs into DSP48E2 slices instead of LUTs, and per-layer
control/glue-logic overhead (the dominant term in why real LUT exceeds this
closed-form model at all) doesn't scale down with bit-width the way raw
arithmetic LUT usage does. A single flat factor (the original version of
this module) applies the avg_bits=8-calibrated multiplier uniformly
regardless of the actual bits chosen — for a low-bit-heavy HAWQ assignment
(the common case: mostly 2/4-bit) that systematically over-penalizes
LUT/BRAM well beyond what real hardware would show, making the ILP overly
conservative exactly where it matters most.

**Model**: linear interpolation of the derating factor between the two real
anchors (avg_bits=3.52 and avg_bits=8 — NOT [2, 8]: there is no real data
point at avg_bits=2), using `avg_bits = (weight_bits + act_bits) / 2` for
whatever unit is being costed, clamped to [3.52, 8] — this model has no
basis to extrapolate below its lower real anchor (a genuinely all-2-bit
assignment would be extrapolating past the measured range, not
interpolating; clamping to the avg_bits=3.52 factor is the conservative
choice). Revisit the moment a real avg_bits<3.52 or a real uniform-low-bit
data point exists.

**Caveat** (now for two points instead of one): one architecture (S19), one
folding regime each (the real build's own resolved PE/SIMD), a
sum-of-independent-partitions build rather than a unified design. Two
points fix the "is this even bit-width-dependent" question (clearly yes)
but not the true curve shape — treat interpolated values as a steering
signal, not a guarantee.

`calibrated_lut`'s `force_dsp=True` switches to the forced-DSP regime's own
flat factor instead (see below); default False preserves prior behavior for
every existing caller.

`calibrated_lut`'s `lut_mult=True` (2026-09-18, additive/opt-in, default
False preserves prior behavior exactly) is for the "hls_lut_noact0"
resource variant specifically: `conv_cost_pe_simd`'s own `mvu_lut` already
applies a real, node-type-specific derate for this regime
(`_HLS_MVU_LUT_MULT_DERATE`, fit on real MVAU_hls+LUT-mult per-node ground
truth) — applying the avg_bits table on top would double-derate with an
unrelated, wrong-direction correction: at avg_bits=8 it multiplies by
8.225x (fit on an older, different real build, before the
noActivation-choice variant framework existed), which taking one real probe
node as an example (MVAU_hls_0, raw=1338, real=953) would inflate the
already-derated 991 prediction to 11,008 — 11.5x too high, not a
correction. So `lut_mult=True` skips this table entirely (identity), the
same way `force_dsp=True` already does, leaving `_HLS_MVU_LUT_MULT_DERATE`
as the only derate applied for this regime.

## Forced-DSP regime (`forced_dsp_lut_total` / `forced_dsp_bram_total`)

2026-09-05: two real S12 8-way-partitioned OOC builds, both forcing DSP on
every MVAU/VVAU node — a different regime from the auto-resType anchors
above (see `compression/hawq/fit_forced_dsp_derating.py` for the full
fitting script and provenance; plan:
`C:\Users\win32\.claude\plans\nested-singing-flurry.md`).

Per-partition real ground truth (8 partitions x 2 builds = 16 points) was
fit two ways:
- **(a) ratio real/raw vs. avg_bits** — broke down badly for partition 0
  (the single `initial.conv`-only partition in both builds: raw_lut=937/1360,
  real_lut=8909/4297, i.e. 9.5x/3.2x, vs. every other partition's
  0.98x-1.86x). Not because partition 0 is expensive in absolute terms
  (it's the cheapest partition in both builds) — because a roughly fixed
  per-partition synthesis overhead (I/O shims, FIFOs, clock/reset infra)
  doesn't shrink with the partition's own logic, so dividing it by an
  equally tiny raw baseline produces a huge ratio. A property of a
  pure-multiplicative model, not of partition 0.
- **(b) real_lut = a\*raw_lut + b (affine, all 16 points, including
  partition 0)**: R²=0.916 — partition 0's residual falls in line with
  everyone else's once the fixed intercept exists to absorb exactly that
  per-partition overhead. Confirms (a)'s diagnosis. Adding total PE/total
  SIMD as extra regressors alongside raw_lut barely moves R² (0.916->0.926,
  with a sign-flipped SIMD coefficient — collinearity noise from 4
  parameters on 16 points) since raw_lut already IS the PE\*SIMD\*bits
  combination; PE/SIMD alone, without raw_lut, fits worse (R²=0.817) than
  raw_lut alone.
- BRAM's affine fit is much noisier (R²=0.462) — real BRAM_18K counts per
  partition are tiny (5-36), dominated by integer-rounding noise at that
  scale; treat BRAM_18K forced-DSP estimates as rougher than LUT's.

The affine model's intercept is a genuine per-partition fixed cost, but
`calibrated_lut`/`calibrated_bram18k` are called per-layer, inside the ILP,
before partition boundaries exist — so the affine form can't be applied
per-layer without over-counting the intercept once per layer instead of
once per partition. Two separate uses instead:
1. Per-layer factor for the ILP's own search (relative cost signal across
   candidate bit assignments): a flat mean factor (not a function of
   avg_bits) computed over the 14 non-degenerate partitions (excluding each
   build's own partition 0). Flat, not sloped, because the real avg_bits
   range across those 14 partitions is only [4.0, 5.15] — S12's min4
   folding convention keeps per-layer bits too tightly clustered for a real
   slope estimate.
2. Affine total-cost check (`forced_dsp_lut_total`/`forced_dsp_bram_total`)
   for validating a concrete partitioned plan (n_partitions known) against
   a hard cap post-hoc: `n_partitions*b + a*raw_total`.

Historical calibration, preserved for provenance/reuse — both real builds
behind these numbers are `12_separable_dense_relu_min4` (S12 separable)
only. Confirmed via real FINN source read (2026-09-16, repo memory
`finn_calibrated_8way_build_status.md`'s "LUT model mismatch: CONFIRMED
root cause" section) that this flat/affine correction was silently
absorbing a whole missing structural term (real FINN's
`addertree_luts`/`acc_luts`, the latter MW-dependent) that `mvu_lut` now
models directly — a factor fit only on separable-architecture (small-MW)
data under-corrects for dense (non-separable, large-MW) architectures
(confirmed empirically: the dense-specific refit's own flat factor, 4.46x,
is ~3.5x larger than separable's 1.26x, even after `mvu_lut`'s structural
fix — see `fit_forced_dsp_derating_s12_dense.py`'s own per-layer folding
comparison: the real dense build's ILP chose far less folding than
separable's did, e.g. `stage2.0.conv` real SIMD=24 vs separable's matched
SIMD=4-6, and this session's re-solved alpha=0.25 output pushes that same
dense layer to SIMD=72, fully unfolded — so a large chunk of the real gap
is a folding-choice/addertree_luts-scaling effect, not pure architecture).
Kept under an architecture-specific name rather than deleted.

Constants: `_S12_SEPARABLE_DSP_FORCED_LUT_FACTOR = 1.261221175430389` (mean
lut_factor, 14 partitions both builds, partition 0 excluded, avg_bits in
[4.0, 5.15]); `_S12_SEPARABLE_DSP_FORCED_BRAM_FACTOR = 0.19434811541501412`
(same 14 partitions); `_S12_SEPARABLE_DSP_FORCED_LUT_AFFINE = (0.947632331261822,
4208.818846016495)` (real_lut = a\*raw_lut + b, all 16 partitions, R²=0.916);
`_S12_SEPARABLE_DSP_FORCED_BRAM_AFFINE = (0.08739251550203067,
8.135659131252611)` (R²=0.462).

### S12 dense (non-separable) refit, 2026-09-15

See `fit_forced_dsp_derating_s12_dense.py` and its own output
`compression/hawq/artifacts/forced_dsp_derating_fit_s12_dense.json`. One
real build (`quantEnet_12_dense_relu_warmstart150ep_alpha025_finn_calibrated_
int8`, 8 partitions, no second build to pool against unlike separable's 16
points) — weaker statistically than the separable fit. avg_bits range
[5.003, 7.284] (this build's own {4,6,8}-candidate joint search, vs
separable's tighter min4 [4.0, 5.15]).
- Flat lut_factor: mean=4.4639 (RMSE=1.6317) over all 8 partitions (unlike
  separable, dense's own partition 0 — avg_bits=6.0, factor=5.24 — is not a
  fixed-overhead outlier the way separable's was, 9.5x/3.2x there vs the
  2.78-6.41 range here for the other 7, so nothing excluded).
- Affine lut fit: a=-1.1204, b=10.9734, R²=0.268 — weak and wrong-signed
  (more bits -> less LUT makes no physical sense) at this sample size; NOT
  used for the active `_FORCED_DSP_LUT_AFFINE` (left at identity) — an
  8-point regression isn't enough evidence to override the post-hoc
  total-cost check with a spurious slope. Stored for provenance only.
- Flat bram_factor: mean=0.4598 (RMSE=0.1245) — real BRAM_18K usage is
  UNDER its own raw prediction here (factor <1), the opposite direction
  from LUT (which needs a >1 correction) — an asymmetry, not a typo.

Constants: `_S12_DENSE_DSP_FORCED_LUT_FACTOR = 4.4639`,
`_S12_DENSE_DSP_FORCED_BRAM_FACTOR = 0.4598`,
`_S12_DENSE_DSP_FORCED_LUT_AFFINE = (-1.1204, 10.9734)` (weak, not applied
by default), `_S12_DENSE_DSP_FORCED_BRAM_AFFINE = (0.1131, -0.1973)` (also
weak at n=8, provenance only).

### Active default: identity (2026-09-17)

Both flat factors (`_FORCED_DSP_LUT_FACTOR`, `_FORCED_DSP_BRAM_FACTOR`) and
both affine constants (`_FORCED_DSP_LUT_AFFINE`, `_FORCED_DSP_BRAM_AFFINE`)
are currently reset to identity (1.0 / (1.0, 0.0)). Every S12 factor above
was fit on builds whose MVAU_hls nodes carried FUSED thresholds
(`outputDataType=UINTx` on 172/173 calibration nodes — the very thing that
made FINN pick HLS over RTL and produced the 4-6x LUT blow-up). The model
now assumes `noActivation=1` + `MVAU_rtl` and prices the standalone
`Thresholding_rtl` explicitly, so those factors no longer describe the
regime being estimated; the six real noActivation/RTL probes in
`hardware/results.csv` sit at 1.1-1.7x of the raw model. Refit against the
RTL production rebuild when it lands; the S12 constants above are kept for
provenance only.

Note for whoever reuses `calibrated_lut(..., force_dsp=True)` for a
non-S12-dense architecture: this is a single global slot, not dispatched by
architecture — applying the dense factor to a future separable-geometry
estimate would over-correct (separable only needs ~1.26x). No per-geometry
selector exists yet; swap the constant back by hand until real dispatch
exists.

## Weight-memory RAM style (`RamStyle` / `RAM_STYLE_BLOCK` / `RAM_STYLE_ULTRA`)

FINN's own "ram_style" nodeattr for the MVU's weight tile (see
`matrixvectoractivation.py`: block=BRAM, ultra=URAM; mutually exclusive,
real FINN's `bram_estimation()`/`uram_estimation()` return 0 for the style
not selected). "distributed" (LUTRAM) is not modeled here — real FINN's
`lut_estimation()` adds an extra c2 LUT term for that style only, which
this closed-form model doesn't (yet) carry; not needed for the block-vs-
ultra BRAM/URAM trade this file supports.

## Impl style (`ImplStyle` / `IMPL_STYLE_HLS` / `IMPL_STYLE_RTL`)

Which FINN backend the MVAU specializes to (`step_specialize_layers`). RTL
is what FINN picks by itself for an MVAU with `noActivation=1`, SIGNED
weights (>= 2 bit) and <= 8-bit operands on DSP48E2 (`_mvu_rtl_possible`);
anything with a fused activation or UNSIGNED weights (this repo's UINT7/
UINT3 weight nodes) stays HLS. `VVAU_rtl` is Versal-only, so depthwise
layers are always HLS on xczu7ev regardless of this setting.

## Standalone Thresholding_rtl (`_thresholding_rtl_cost`)

Empirical per-node cost, a real function of both `pe` AND `numSteps`
(2026-09-17 refit — the previous pe-only/flat-BRAM version is kept only in
git history). `numSteps = 2**output_bits - 1` (number of real threshold
values FINN's `Thresholding_rtl` actually stores) = a direct structural
fact of threshold-based quantization (to sort a value into one of 2^B
output levels you compare against 2^B-1 sorted thresholds, see
`thresholding.sv`'s own binary-search pipeline — not a fitted
relationship). `output_bits` here is this layer's own `act_bits` (the ILP's
`y[layer,w,a]` choice) as the best available proxy for the real
`outputDataType` bitwidth `Thresholding_rtl` actually stores — same
approximation used elsewhere in this file for the fused-threshold case (no
separate output-precision axis exists).

**Refit 2026-09-17** on 168 real Thresholding_rtl nodes
(`hardware/datasets/mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_
alpha025_rtl_mvau_noact1_extended.csv` — the real S12-dense-warmstart
alpha=0.25 8-way production build), TRAIN split;
`hardware/datasets/mvau_variant_matrix_dataset_extended.csv`'s
`noact1_auto_*` rows (n=18, but PE=1/numSteps=255 for every single row —
zero variance in either axis, so this slice cannot validate a PE or
numSteps dependence, only the overall level at that one point) held out as
TEST — not a proper held-out validation of this specific relationship,
flagged honestly.

The old flat/pe-only model was confirmed wrong, not just imprecise: real
BRAM18 at PE=1/numSteps=15 (4-bit output) averages 0.05 (n=20) — the old
flat `_THR_RTL_BRAM18_PER_NODE=3.6835` constant overshoots that ~74x —
while PE=8/numSteps=255 (8-bit) averages 19.0 — the same flat constant
undershoots that ~5x. Both LUT and BRAM needed numSteps added, not just PE:
- **LUT**: `real_LUT ~= PE * (_THR_RTL_LUT_BASE_PER_PE +
  _THR_RTL_LUT_PER_NUMSTEP_PE * numSteps)` — PE-only R²=0.950; adding the
  numSteps term -> R²=0.960. Physically: a fixed per-PE-lane
  comparator/control cost (independent of how many thresholds) plus a
  per-PE-lane term that scales with numSteps (each PE lane's own
  LUTRAM-packed threshold set). OLS through origin on both terms (no free
  intercept, same convention as every other constant in this file).
- **BRAM18**: `real_BRAM18 ~= _THR_RTL_BRAM18_PER_PE_NUMSTEP * PE *
  numSteps` — PE-only R²=0.707; numSteps-only R²=0.232; the pure
  interaction term alone (no separate PE or numSteps terms) gets R²=0.812,
  actually beating every free-intercept multi-term model tried (unstable,
  sign-flipping coefficients from PE/numSteps collinearity in this dataset
  — 4 PE values x 3 numSteps values, not a full factorial). A single clean
  multiplicative constant generalizes more honestly than an overfit 3-4
  parameter model here.

Constants: `_THR_RTL_LUT_BASE_PER_PE = 70.6827`,
`_THR_RTL_LUT_PER_NUMSTEP_PE = 0.1065`,
`_THR_RTL_BRAM18_PER_PE_NUMSTEP = 0.011444`.

### `_THR_RTL_URAM_PER_PE_NUMSTEP` — derived, not fit

`real_URAM==0` for all 168/168 Thresholding_rtl nodes in the training data
(`depth_trigger_uram=0` throughout, FINN's own per-stage min-waste
primitive selection — `finn.util.basic.mem_primitives_versal`/
`get_memutil_alternatives`, used unconditionally regardless of target part
despite the "versal" name) — never once favored URAM for this network's own
geometries. So there is no real per-node ratio to fit; this constant is
instead scaled from the real, validated BRAM18 constant by the fixed,
device-family-independent capacity ratio of the two real primitives: BRAM18
= 18Kib = 18,432 bits (the "36x512" shape in `mem_primitives_versal`) vs
URAM288 = 288Kib = 294,912 bits (the native "72x4096" shape) — exactly 16x
denser per block. `_THR_RTL_URAM_PER_PE_NUMSTEP =
_THR_RTL_BRAM18_PER_PE_NUMSTEP * (18_432 / 294_912)` = 0.0007153.

This was briefly a real, free per-layer BRAM-vs-URAM choice in
`finn_milp.py`'s ILP (2026-09-17/18), to trade BRAM pressure onto idle
URAM — **retired 2026-09-18**: a real attempted Vivado synthesis of
Thresholding_rtl with `ram_style="ultra"` FAILS outright. URAM288 cannot be
used as ROM, and Thresholding's memory is exactly that: a compile-time-
constant lookup table (the threshold values), never written at runtime —
URAM lacks the INIT-file-based ROM initialization mechanism BRAM18/36
primitives have. `finn_milp.py` now hard-fixes `thr_ram_style="block"`
unconditionally; this constant and the `ram_style="ultra"` branch are kept
for provenance and any future diagnostic/what-if use, but are no longer
reachable from any live ILP run.

### `_THR_RTL_LUTRAM_PER_PE_NUMSTEP` — derived, not fit (2026-09-18)

Same derivation method as the URAM constant, landing somewhere genuinely
usable this time. Unlike `ram_style="ultra"` (real Vivado synthesis fails
outright), "distributed" IS structurally real for Thresholding_rtl:
`thresholding.sv`'s own `RAM_STYLE` localparam has a genuine
"auto"/"distributed" choice (confirmed via direct .sv source read),
forceable per-node by setting `depth_trigger_bram` above the node's own
real depth. It's just never been exercised in any real build (every real
node so far used "auto", i.e. Vivado's own choice, which always landed on
BRAM for the geometries tested — 0/240 real Thresholding_rtl nodes across
both datasets ever showed nonzero real_LUTRAM). So: no real per-node ratio
to fit here either, same situation as URAM. Scaled from the same real,
validated BRAM18 constant, but by the real capacity ratio to FINN's own
LUTRAM primitive shape instead of URAM288's: `finn.util.basic.
mem_primitives_versal` defines `"LUTRAM": (1, 64)` — width=1 bit, depth=64
words, i.e. 64 bits per primitive (a real, standard Xilinx shape: one LUT6
configured as RAM64X1S distributed RAM, not an invented number). BRAM18 =
18,432 bits per block is 288x denser than one 64-bit LUTRAM primitive, so
the same structural assumption as the URAM derivation gives a much larger
per-(pe,numStep) coefficient than URAM's — and correctly so: for this
project's real numSteps range (commonly 255), distributed genuinely IS a
bad deal (predicts ~840 LUTs for a numSteps=255 node needing only ~3
BRAM18 blocks), which is exactly why Vivado's own "auto" heuristic has
never once picked it in any real build. It only becomes competitive at
small numSteps, where BRAM's own fixed per-block overhead dominates —
exactly the resource-pressure tradeoff an ILP is suited to evaluate per
layer, unlike a fixed local rule.

`_THR_RTL_LUTRAM_PER_PE_NUMSTEP = _THR_RTL_BRAM18_PER_PE_NUMSTEP * (18_432 /
64)` = 3.295872.

## Residual-join thresholds (`threshold_node_cost`, 2026-09-26)

Until 2026-09-26 the cost model only priced thresholds that follow a conv
(each conv's own `thr_*` under `noActivation=1`). Decoding the checked-in
partition graphs of the real current-regime build
(`hardware/outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349/intermediate_models/supported_op_partitions/partition_*.onnx`,
MVAU_rtl, `noActivation=1`) shows every residual block has extra standalone
`Thresholding_rtl` nodes around its `AddStreams_hls` that nothing priced.
A regular block:

```
Add_0 → Thr (INT8) → Thr (UINT8) → Dup ─┬─ Thr (INT8) ─────────────────────────────────────────────┐
                                         └─ MVAU reduce → Thr → MVAU 3x3 → Thr → MVAU expand → Thr ─ Add_1
```

| node | what it is | output type in the build |
|---|---|---|
| `<block>.skip_quant` | `QuantEltwiseAdd.input_quant` requantizing the skip operand | always INT8 |
| `<block>.residual_add` | the add's own output quant | always INT8 (even where `out_act` is UINT4) |
| `<block>.out_act` | the block-final activation | follows the per-site HAWQ bits |

- The main operand's requant is NOT a separate node: `expand`'s own
  threshold absorbs BN + the add's `input_quant` (it outputs INT8 straight
  into the add) and is already priced as `expand.0`'s `thr_*`.
- `skip_quant` only exists when the skip path has no conv of its own.
  Regular blocks: skip is the previous block's `out_act` → separate node.
  Upsampling blocks: skip ends in `main_proj`/`skip_resize_conv` → that
  conv's threshold absorbs it. Downsampling blocks: FINN lowers the
  zero-pad `torch.cat` on the pool path into a 1x1 MVAU (e.g. MH=32, MW=16)
  followed by the INT8 requant — `skip_quant` stands in for that threshold;
  the pad-MVAU itself is **not modeled** (it isn't in the FP32 graph).
- Also unpriced and not modeled: `AddStreams_hls`, `DuplicateStreams_hls`,
  the downsampling pad-MVAU, and InitialBlock's extra thresholds (input
  quant, pool-path requant, post-concat act).

The older separable build (`hawq_12_sep_…`, fused `noActivation=0` MVAU_hls)
has the same skip-branch threshold after the fork, but a different
arrangement of the rest — only the current-regime build was used here.

`threshold_node_cost` reuses `_thresholding_rtl_cost` (the per-node fit),
with `cycles = hout*wout*ceil(cout/PE)` — a per-channel compare with no
reduction axis. The fit's BRAM term scales with PE×numSteps and ignores the
channel count; at the fixed 8-bit (numSteps=255) that is ~2.9 BRAM18 per PE
lane, so these nodes are a real BRAM consumer, not a rounding error.

## `LayerGeometry.groups` / depthwise fix

`groups=1` (default) is byte-for-byte identical to this field not existing.
`groups=cin=cout` is a true depthwise conv (see `ENet.py`'s
`DSCNoProjectionBottleneck`/`RegularBottleneck`'s `use_dsc` branch): each
output channel reduces over only `cin/groups` input channels, not the full
`cin` — previously silently treated as a dense conv (`Q=cin*kh*kw` instead
of the real `(cin/groups)*kh*kw`), overstating LUT/BRAM/PE/SIMD by a factor
of ~groups for every depthwise layer in any DSC/dsc_no_projection
architecture (S8/S10/S13/S15/S16/S19-DSC variants, `22_dsc_projected`).
Does NOT affect `swu_bram18` (the sliding-window buffer still holds all cin
input channels' worth of pixels regardless of grouping) or any architecture
that never sets `groups>1` (e.g. the `26_9_w24_s14w12_nonneg_block` family,
which uses `separable_dilated`'s (k,1)+(1,k) dense factoring, not grouped
convs).

`is_depthwise` asserts `groups == cin == cout` defensively (partial-group
convs aren't a supported VVAU shape and no config in this repo ever
produces one) — the shape that becomes a FINN VVAU node instead of an MVAU,
with its own PE/SIMD folding domain and preceding SWU/FMPadding coupling
constraint (`compression/hawq/folding_ilp.py`'s `solve_folding_nodewise`).

`swu_max_simd_depthwise`: the preceding ConvolutionInputGenerator's (SWU's)
own SIMD must divide `IFMChannels` — for a depthwise layer that's
`cin==cout==groups`, NOT `cin*kh*kw` (that's the MVAU-SIMD constraint for a
dense conv, a different axis — see `hardware/finn_native_cost_
estimator.py`'s own module docstring for the real-FINN-source-derived note
on this mismatch). Deliberately the same domain as this layer's own VVAU PE
(`max_pe(layer)==layer.cout`) — `solve_folding_nodewise`'s coupling
constraint requires an SWU SIMD choice and a VVAU PE choice to be able to
agree exactly.

## `MEM_MODE_DECOUPLED`

`"internal_decoupled"` — real FINN v0.10.1 `mem_mode` nodeattr value for
MVAU/VVAU nodes whose weights are streamed from BRAM/URAM rather than
embedded as LUT/FF constants (confirmed against a real generated
`auto_folding_config.json` — the informal name "decoupled" is this
version's stale/pre-rename name for it).

## SWU / ConvolutionInputGenerator_rtl (`_finn_swu`)

Returns `(swu_lut, swu_bram18, swu_uram18, swu_cycles)`.

**A 1x1 kernel gets NO SWU NODE AT ALL** — confirmed structurally, not
numerically: across every real node in
`hardware/mvau_swu_threshold_calibration_dataset.csv` (production
S12-separable partitions + the int6_pemh probe, whose exact per-layer
kernel sizes are known), MVAU (MH,MW) pairs where `MW==IFMChannels`
(kh=kw=1, e.g. every ENet bottleneck's 1x1 reduce/expand) are NEVER
immediately preceded by a `ConvolutionInputGenerator_rtl` node (38/38
occurrences); pairs with a real kernel area (MW a multiple of IFMChannels,
e.g. 3x1/1x3/3x3) always are (35/35). A 1x1 conv reads one input pixel per
output pixel with nothing to buffer, so FINN's own `InferConvInpGen`
transform simply doesn't insert the node — a graph-structure fact, not
something `get_buffer_depth()`'s formula (which would still return a small
nonzero value at kh=kw=1) would tell you.

For a real kernel (kh,kw not both 1): BRAM and cycles depend on the SWU's
own SIMD through `channel_factor = IFMChannels/SIMD` — a small SIMD means a
deep, narrow line buffer (worse RAMB18 aspect ratio) and proportionally
more cycles per output row — this is where FINN's "unfold SIMD before PE"
guidance is actually grounded. `impl_style` follows `select_impl_style()`:
"parallel" for `parallel_window=1` (window emitted whole, `cycles = number
of input words + 2`), else "default" (`get_buffer_depth()`/
`get_exp_cycles()`'s 2D branch). The 1D-input branch (`ifm_dim_h==1 or
ifm_dim_w==1`) is not reproduced — no layer in this repo has a
1-pixel-high/wide input. No real `parallel_window=1` node exists yet in the
calibration data (all 43 are `impl_style="default"`) to confirm
`get_buffer_depth()`'s own "parallel" formula (used here) — NOT
`bram_estimation()`'s separate inline "parallel" buffer_depth
(`kernel_width/(win-kernel_width)` form, used only for that function's own
BRAM sizing, and easy to confuse with this one — an earlier version of this
function did exactly that).

### `ram_style` for SWU (2026-09-17 addition)

Default `"distributed"` preserves this function's exact prior behavior.
FINN's `ConvolutionInputGenerator_rtl.lut_estimation()` adds a LUTRAM term
(`buffer_width*ceil(buffer_depth/38)`) only for `ram_style="distributed"`
(the default); `bram_estimation()` only counts BRAM for "block"/"auto";
`uram_estimation()` only counts URAM for "ultra" — all three mutually
exclusive, transcribed directly from FINN's own three functions.

A fourth value, `"auto_efficient"` (NOT a real FINN ram_style — resolved
internally), computes BOTH `_finn_buffer_bram18` and `_finn_buffer_uram18`
for this exact (buffer_width, buffer_depth) and deterministically keeps
whichever needs fewer physical blocks (ties go to block) — a genuine
per-buffer efficiency comparison, not real FINN's own "auto" (which never
considers URAM at all, same blind spot as MVAU's own `ram_style="auto"`).
This is what `finn_milp.py`'s ILP passed for every layer briefly
(2026-09-17), before being retired the very next day (2026-09-18): by then
48/48 real SWU nodes across both real datasets had used
`ram_style="distributed"` with `real_BRAM18==real_URAM==0`, regardless of
what ram_style the paired MVAU's own weight memory used — and for the
actual deployed geometry this "auto_efficient" mode predicted a
non-trivial 41 URAM blocks (42.7% of the 96-block budget), a real,
consequential-sized claim with zero supporting evidence. `finn_milp.py` now
hard-fixes SWU to `swu_ram_style="distributed"` instead. This mode is kept
for provenance/what-if use, but is no longer reachable from any live ILP
run. Both `_finn_buffer_bram18`/`_finn_buffer_uram18` remain direct
FINN-source transcriptions with no real "block"/"ultra" SWU ground truth
anywhere — this auto-pick was only ever as trustworthy as those two
formulas are, which is to say: plausible, and now empirically shown to not
match real hardware.

### `_finn_buffer_bram18` / `_finn_buffer_uram18`

`_finn_buffer_bram18`: FINN `ConvolutionInputGenerator_rtl.bram_estimation()`'s
RAMB18 count for ONE line buffer (ram_style block/auto): aspect ratio
chosen by depth, cascaded past 16384 words, with FINN's own
remainder-cascade saving.

`_finn_buffer_uram18`: FINN `ConvolutionInputGenerator_rtl.uram_estimation()`'s
URAM288 count for ONE line buffer (ram_style="ultra"): fixed 4096-deep x
72-wide aspect ratio (real URAM288 shape, same constants MVAU's own
`uram_estimation()` uses for weight memory) — no aspect-ratio table or
cascade-remainder saving the way BRAM has, since URAM only comes in this
one shape. Uncalibrated: every real SWU node checked in this repo's
calibration data (43/43) used `ram_style="distributed"` — there is no real
`ram_style="ultra"` SWU ground truth yet, so this is a direct FINN-source
transcription, not a fitted/verified formula (same status as
`_finn_buffer_bram18`, equally untested against real "block" SWU data).

### `_SWU_LUT_DERATE = 0.755`

Provisional, OLS-through-origin fit, n=43 real `ConvolutionInputGenerator_rtl`
nodes (`hardware/mvau_swu_threshold_calibration_dataset.csv`) — real Vivado
LUT comes in 51-104% of this formula's estimate (mean 70%), same direction
as `_RTL_MVU_LUT_DERATE` (real RTL synthesis under FINN's own analytical
estimate). Refit once the full production rebuild lands.

## `conv_cost_pe_simd` — the general per-layer cost

The general per-layer cost, given explicit PE/SIMD (the actual folding
decision variables a folding search chooses over) instead of just the two
folding-preset endpoints. `conv_cost`/`_pq_for_folding` are thin wrappers
around this for the two presets used elsewhere (per-stage HAWQ bit-width
search); a folding ILP wants the full (PE, SIMD) space, not just those two
points.

`cycles ~= ceil(H_out*W_out/M) * ceil(C_out/PE) * ceil(Q_max/SIMD)` — FINN's
own analytical per-layer cycle estimate (same category as its real
`estimate_layer_cycles.json` report): each of the H_out*W_out output pixels
needs one pass per PE-group of output channels times one pass per
SIMD-group of the reduction. Using `ceil()` rather than requiring PE/SIMD to
be exact divisors keeps this usable for arbitrary values, though every
caller in this codebase only ever passes divisors (see `divisors()`), where
ceil reduces to exact division anyway.

### `ram_style` (weight tile)

Mirrors real FINN's mutually-exclusive `bram_estimation()`/
`uram_estimation()`: "block" (default, unchanged behavior) puts weights in
BRAM, `wm_uram18=0`; "ultra" puts them in URAM instead (`wm_uram18 =
ceil(mem_width/72) * ceil(omega/4096)`, the exact formula real FINN's
`uram_estimation()` uses), `wm_bram18=0`. LUT cost is identical either way
(confirmed via direct FINN source read: real FINN's `lut_estimation()` only
adds an extra term for `ram_style="distributed"`, not "ultra") — URAM is a
free swap in this cost model other than needing its own separate resource
budget.

### `swu_ram_style` / `thr_ram_style`

`swu_ram_style` (2026-09-17, default "distributed" preserves exact prior
behavior) is a separate choice for the SWU's own line-buffer memory — a
different physical memory from the weight tile `ram_style`. Deliberately
NOT tied to `ram_style`: real calibration data (43/43 real SWU nodes at
`ram_style="distributed"` regardless of the paired MVAU's own ram_style)
rules out coupling them. `finn_milp.py`'s ILP passes `"auto_efficient"`
here for every layer (a deterministic, geometry-driven pick, not a free ILP
variable) rather than a fixed style.

`thr_ram_style` (2026-09-17, default "block" preserves exact prior
behavior) is a third, independent memory choice — for the standalone
Thresholding_rtl node's own memory, a different physical memory again from
both the weight tile and the SWU line buffer. Unlike `swu_ram_style`, this
one IS a real, free per-layer ILP choice in `finn_milp.py` (not a
deterministic "auto_efficient" pick): FINN's own local per-primitive
min-waste selection never favors URAM here, but that's the wrong criterion
for a design where BRAM is the globally scarce resource and URAM sits
mostly idle — a genuine resource-tradeoff decision the ILP's own joint
optimization is better positioned to make than any local per-node
heuristic.

### `force_dsp` / `impl_style`

`force_dsp` mirrors real FINN's `resType="dsp"` nodeattr (see
`matrixvectoractivation_hls.py`'s `lut_estimation()`): zeroes `mvu_lut`'s
LUT-based multiplier term (`mult_luts`) since the multiplication moves to
DSP48 slices instead — pass the same `force_dsp` used elsewhere for this
build (e.g. the one already threaded into `calibrated_lut`/
`calibrated_bram18k`) for consistency; the two used to disagree (raw
`mvu_lut` always included a multiplier term regardless of force_dsp, only
the separate calibration factor knew about it).

`impl_style` (default RTL, the go-forward build target) only changes the
DSP count (FINN's `MVAU_rtl.lut_estimation()` is 0 — there is no RTL LUT
model, the HLS one is used as the proxy for both) and implies DSP
multipliers. Depthwise layers are forced to HLS (`VVAU_rtl` is
Versal-only). `act_signed` feeds FINN's accumulator-width bound (this
repo's post-ReLU activations are UINT).

### `no_activation` (2026-09-17 addition, additive/opt-in)

Every conv layer defaults to `noActivation=1` (`no_activation=True`), i.e.
a separate Thresholding_rtl node (PE assumed == this layer's PE) priced
empirically as `thr_lut`/`thr_bram18`/`thr_uram18` and folded into
`total_lut`. `cycles = max(MVAU cycles, SWU cycles)`: a layer runs at the
speed of its slowest node.

`no_activation=False` models FINN's other real regime: the
activation/threshold is fused into this MVAU/VVAU node itself
(`noActivation=0` in real FINN's own nodeattr) instead of living in a
separate standalone node. Only legal under the HLS backend
(`impl_style="hls"`) — `MVAU_rtl`/`VVAU_rtl` structurally require
`noActivation=1` (see `finn_milp.py`'s own eligibility filter for the ILP's
"rtl_dsp_noact1" vs "hls_lut_noact0" variants); this function itself does
NOT enforce that — calling it with an illegal combination just produces a
number nothing in real FINN could build.

When `no_activation=False`: `thr_lut`/`thr_bram18`/`thr_uram18` are all 0
(no separate node exists any more), and `thr_pe` is 0 (meaningless).
Instead, real FINN's own `MVAU_hls.lut_estimation()` (transcribed directly)
adds a "thr_luts + comp_luts" term inside this node's own `mvu_lut`, gated
on `ram_style_thresholds` (default "auto"): these terms are literally 0
unless `ram_style_thresholds=="distributed"` — FINN's own static resource
estimator has NO modeled LUT/BRAM cost for fused thresholds under its own
default ("auto") or "block" styles (Vivado decides at synthesis time;
FINN's own analytical model just doesn't price it). A real, confirmed
blind spot in FINN's own estimator. So at the default
`ram_style_thresholds="auto"`, `no_activation=False` is modeled as pure
savings (the standalone node's cost disappears, nothing added in its
place). Only pass `ram_style_thresholds="distributed"` to exercise the real
fused-LUT term; `B` (the output activation bit-width FINN's own formula
uses) is approximated as this layer's own `act_bits` — a real
approximation, not a FINN-source transcription, and untested against
hardware.

### `mvu_lut` breakdown

Real FINN's `MatrixVectorActivation_hls`/`VectorVectorActivation_hls.
lut_estimation()` (Blott et al. FINN-R, confirmed via container source read
2026-09-16) — `c0 + c1*P*(mult_luts + addertree_luts + acc_luts)` — NOT the
old flat "300 + 1.1\*P\*Q\*W\*A" stand-in. `mult_luts` is the LUT-based
multiplier array, zero when `force_dsp` (multiplication moves into DSP48
slices instead); `addertree_luts` is the SIMD-wide reduction tree;
`acc_luts` is the accumulator register width, which grows with `log2(MW)`
where MW is the full reduction depth (`max_simd(layer)` — the whole
`cin/groups*kh*kw`, NOT the folded Q) — a real dependence on kernel/channel
geometry the old flat formula had no way to express (root cause of the
dense-vs-separable LUT under-prediction, see repo memory
`finn_calibrated_8way_build_status.md`'s "LUT model mismatch: CONFIRMED
root cause" section). `thr_luts`/`comp_luts` (fused-threshold LUTs,
`(2^B-1)*acc_bits` per PE) are 0.0 whenever `no_activation=True` (the
module default) or `ram_style_thresholds != "distributed"` (also the
default). Note pre-2026-09-17 production builds did NOT satisfy that:
172/173 nodes in `hardware/mvau_lut_calibration_dataset*.csv` have
`outputDataType=UINTx`, i.e. fused thresholds -> forced MVAU_hls (RTL needs
noActivation=1) and a `PE*2^B`-scaled LUT blow-up that the former
"imbalance_luts" term was chasing empirically. Standalone-threshold builds
are priced via `thr_lut` instead.

The accumulator-width bound: FINN's `alpha = log2(MW) + W + A - 1 -
int(idt.signed())`, `acc_bits = min(accDataType width, ceil(alpha +
log2(1+2^-alpha) + 1))` — the https://arxiv.org/abs/2301.13376 bound,
capped at the INT32 default. This repo's post-ReLU activations are
unsigned (165/173 calibration nodes UINT4/6/8), hence `act_signed=False` by
default.

The former empirical "imbalance_luts" term — `(k_pe*PE/SIMD +
k_mh*max(0,MH-mw))*A`, pooled R²=0.779 on the two fused-threshold
calibration sets — was removed 2026-09-17. It was a proxy for FINN's own
`thr_luts`/`comp_luts`: LUT/PE in that data scales ~3x per +2 output bits,
every "SIMD=1 cliff" row had 8-bit outputs, and geometry-identical
MVAU_rtl/noActivation probes show none of it. See
`analysis/hardware_calibration/` for the exploration.

FINN-R constants (`MVAU_hls`/`VVAU_hls.lut_estimation()`): `c0=300, c1=1.1`.
`c2`, FINN's extra LUTRAM term, only applies to `ram_style="distributed"`
(or embedded weights <= 128 words) — neither is selectable here
(block/ultra), so 0.

### RTL LUT derating, `_RTL_MVU_LUT_DERATE = 0.4868`

`MVAU_rtl.lut_estimation()` is literally `return 0` in FINN v1.0.0-alpha —
there is no real RTL LUT model, so the HLS formula above is used as a
stand-in for the RTL backend too. `real_LUT` (the MVAU_rtl node alone, per
its own Vivado hierarchy row — NOT swu_lut/thr_lut, genuinely separate FINN
graph nodes) comes in consistently lower than this formula, across every
node regardless of MH/PE. Makes physical sense: MVAU_rtl's `add_multi.sv`
folds accumulation into the DSP58 cascade chain, doing in hardware what
`addertree_luts`/`acc_luts` assume needs separate LUT logic.

First fit (2026-09-17, briefly): n=7, one probe
(`hardware/mvau_lut_calibration_dataset_s12_context_dense_int6_pemh_simd1.csv`),
one architecture, one folding pattern (PE=MH/SIMD=1, W6A6) — OLS-through-
origin gave 0.581, R²=0.747, explicitly flagged provisional pending the
full production rebuild.

**Refit (2026-09-18)**: that production rebuild now exists
(`hardware/datasets/mvau_lut_calibration_dataset_12_dense_relu_warmstart150ep_
alpha025_rtl_mvau_noact1_extended.csv`, 88 real MVAU_rtl nodes, many (PE,
SIMD, weight_bits, act_bits, MW) combinations from the real 8-way deployed
build) — refitting on it (same OLS-through-origin convention) gives 0.4868,
R²=0.887 (up from 0.747 when the old 0.581 constant is scored against this
same larger dataset) — a tighter fit on far more data, not a different
formula. At the aggregate level: this build's 88 real MVAU_rtl nodes sum to
59,630 real LUT; the old constant (0.581) predicted 71,072 (+19%), the
refit constant (0.4868) predicts 59,552 (+0.1%). Only applies to
`impl_style="rtl"` — VVAU_hls (depthwise) stays unscaled (no real depthwise
MVAU_rtl data exists yet either way, since VVAU_rtl needs Versal/DSP58 and
this repo targets xczu7ev); the non-depthwise `impl_style="hls"+use_dsp=False`
(LUT-mult) case gets its own separate derate instead (below), since it's a
structurally different regime (MVAU_hls.lut_estimation() is a real formula,
unlike MVAU_rtl's `return 0`, so there was never a reason to assume the
same bias applies).

### HLS LUT-mult derating, `_HLS_MVU_LUT_MULT_DERATE = 0.7402` (2026-09-18)

`MVAU_hls.lut_estimation()` (this formula) is a real FINN model, unlike
MVAU_rtl's `return 0` stand-in above — but it's still an analytical
estimate, and real synthesis diverges from it too. First real per-node
ground truth (`hardware/datasets/mvau_variant_matrix_dataset_extended.csv`,
the `noact1_hls_lut_*` combos: standalone Thresholding_rtl + MVAU_hls
forced `resType=lut`, 3 real conv layers (MH/MW = 8/32, 8/72, 32/8) x 3
fold strategies (pemh_simd1/balanced/pe1_simdmw) = 9 real MVAU_hls nodes,
W8A8): real_LUT comes in at 0.83-1.48x of this formula's raw prediction
(OLS-through-origin fit 0.7402, R²=0.885) — consistently lower, same
direction/rough magnitude as the RTL derate above, though this is a
genuinely separate fit (different node type, different formula branch:
`mult_luts` is nonzero here, 0 for RTL). Provisional: n=9, one probe
network, one bit-width (W8A8), PE/SIMD in {1,8,32,72} only — refit if/when
a production hls_lut_noact0 build exists. Only applies when `use_dsp=False`
(the actual LUT-mult case) — `impl_style="hls"` with `use_dsp=True` (the
`VARIANT_HLS_DSP_NOACT0` placeholder, never live in the ILP) stays
unscaled, no real motivating data checked for it yet.

### Standalone Thresholding node's PE (`thr_pe`)

The standalone Thresholding_rtl that consumes this layer's accumulators
under `noActivation=1`. Its PE is NOT a free choice and NOT simply == the
MVAU's PE: the MVAU emits one fold of P outputs every `SF = MW/Q` cycles,
so per output pixel the threshold node has `NF*SF = (MH/P)*(MW/Q)` cycles
to process MH channels at `PE_thr` per cycle -> it keeps up iff `PE_thr >=
P*Q/MW` (and FINN requires `PE_thr | NumChannels`). The smallest such
divisor is what SetFolding-style balancing would pick; anything larger only
costs LUT. Note for the build side: FINN's default Thresholding PE is 1,
and the folding configs carry no Thresholding entries — the bridge must
write `thr_pe` explicitly, or a PE=1 node throttles any layer whose MVAU
needs `PE_thr > 1`.

`no_activation=False`: no standalone Thresholding node exists in the graph
at all (the activation is fused into the MVAU/VVAU node instead) — its
resources are simply gone, not moved, so
`thr_lut/thr_bram18/thr_uram18=0` and `thr_pe=0` (meaningless, no such node
to give a PE to).

### DSP formula

HLS: `MVAU_hls.dsp_estimation()` `P*Q*ceil((W+A)/48)` (one DSP48E2 per MAC
lane; VVAU_hls's own formula is `P*ceil((W+A)/48)`, no SIMD factor —
transcribed as-is), zero unless the multiply is on DSP.

RTL: FINN's own `MVAU_rtl.dsp_estimation()` source
(`util.basic.get_dsp_block(fpgapart)=="DSP58"` -> `P*ceil(Q/3)`; else
(DSP48E1/E2, which is what xczu7ev-ffvc1156-2-e resolves to) ->
`ceil(P/4)*Q`, no bit-width dependence at all) says 4 PE lanes per DSP48E2,
unconditional on W/A. Real per-node ground truth
(`hardware/probes/mvau_lut_calibration_dataset_s12_context_dense_int6_pemh_
simd1.csv`, 7 real MVAU_rtl nodes, W6A6) shows `real_DSP = ceil(P/2)*Q`
exactly on every node (16/4/4/16/4/4/16) — exactly 2x FINN's own formula,
at every (P,Q) pair tested. A previous version of this code branched on
bit-width (4 lanes for <=4-bit, 2 for >4-bit) based on the 8-bit pemh
aggregate probes also landing at 2x — that branch had no real evidence
behind its <=4-bit half: every <=4-bit data point available
(dense_int4/separable_int4, PE=1 throughout) can't distinguish `ceil(1/4)`
from `ceil(1/2)` (both round up to 1), so "4 lanes at <=4-bit" was an
untested assumption, not a finding. Corrected 2026-09-17 to a flat,
unconditional 2 lanes/DSP — the only value with real per-node support at
high PE, and consistent (if uninformative) at PE=1 too. Refit if a real
high-PE low-bit (e.g. int4_pemh) probe ever exists.

## `conv_cost` / `conv_transpose_cost` / `maxpool_cost`

`conv_cost`: Conv2d cost (Eq. 1/4/5 of `finn_cost_formulae.md`'s source
paper) at one of the two folding presets — `conv_cost_pe_simd` for the
general (arbitrary PE, SIMD) version a folding search needs.
`ConvTranspose2d` is handled by the caller pre-converting its geometry into
the equivalent zero-inserted dense conv (`conv_transpose_cost`) before
calling this. Uses the general BRAM_wm formula (`omega =
K^2*C*C'/(Q*P)`), not a fixed `omega=1` shortcut — at `FOLDING_UNFOLDED`
this still simplifies to omega=1 automatically, so those numbers are
unchanged/still verified; at `FOLDING_SERIAL` (Q=P=1) omega is the full
weight volume, correctly reflecting "the whole layer's weights get streamed
through one PE over many cycles" instead of loaded all at once.

`conv_transpose_cost`: ConvTranspose2d modeled as zero-insertion + ordinary
stride-1 conv (Dumoulin & Visin) — see `finn_cost_formulae.md`'s own
derivation. Only `K=S, p=0` transposed convs are used anywhere in this
architecture (`up4.up.0`, `up5.up.0`, `final`), matching that file's own
confirmed case.

`maxpool_cost`: MaxPool2d = SWU + comparator array, no MVAU/weights —
`act_bits` only (no `weight_bits`, nothing to quantize). No P/Q/folding at
all — pooling was never a folded MVAU to begin with. `cycles`: real FINN's
`StreamingMaxPool.get_exp_cycles()` (`hls/streamingmaxpool_hls.py`) is
`int(ifm_dim**2 * (1 + 1/k**2))` — input-pixel-driven, NOT hout*wout. A
pooling window still has to read every input pixel once regardless of how
many output pixels the stride produces, so cycles scale with `hin*win` —
~k²x more than the output-pixel count this formula previously (wrongly)
used, which silently assumed downsampling also cuts cycle cost by the same
factor (confirmed a real ~4x undercount for this repo's 2x2/stride-2
pools).

## `layer_cost_pe_simd_auto_ram`

Like `layer_cost_pe_simd`, but picks `ram_style` per-layer instead of
taking it as a fixed input — the "leave memory type as auto" default for
cost-model estimates going forward (standing convention, 2026-09-15). LUT
is identical between "block" and "ultra" (real FINN's `lut_estimation()`
only adds an extra term for `ram_style="distributed"`, not "ultra") — so
"auto" only ever changes which pool (BRAM_18K vs URAM) a layer's weight
memory lands in, never total_lut/cycles. Picks whichever of
`wm_bram18`/`wm_uram18` is the smaller block count for this layer (a
stand-in for real FINN's own auto ram_style heuristic, not directly
accessible from this repo). MaxPool2d has no weight memory at all —
`ram_style` is irrelevant there, both calls would return identical results
anyway, so it's skipped.

**Per feedback memory** (`finn_cost_model_force_dsp_auto_ram.md`): new
cost-model calls should use `layer_cost_pe_simd_auto_ram(..., force_dsp=True)`,
not a hardcoded block/ultra `ram_style`.
