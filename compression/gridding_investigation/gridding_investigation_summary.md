# Gridding Artifact Investigation — Summary

Model under test: **S5-DscNoProjDense** (`nnUNetTrainerENet_5_1_dscnoprojection_dense_dilation`,
channels 4,16,32,16,4, `dsc_no_projection=1`, `context_pattern=dense_dilation`) — our
best-Dice compressed config (0.8105), matching the "ENet U4, DSC bottlenecks, no 1x1
projection" description exactly. Its stage2/3 dilation schedule is (2,4,8,16) repeated
twice over 8 blocks, per `DENSE_DILATION_PATTERN` in `enet/nnunetv2/nets/ENet.py`.

Scripts: `task1_structural_test.py`, `task2_empirical_test.py` (both under this folder).
Raw outputs: `task1_structural_results.json`, `task2_empirical_results.json`, and PNGs in
`results/`.

## 1. Task 1 — structural (impulse-response) coverage

Real stage2/3 resolution (measured via forward hook on the actual model, 512x512 input):
**64x64** → cumulative downsample factor **8x**. All three schedules sum to 30, giving an
identical theoretical receptive field of 61x61 for all three.

| Schedule | Rates | RF size | Coverage ratio | Heatmap |
|---|---|---|---|---|
| Current | 2, 4, 8, 16 | 61x61 | **25.8%** | `gridding_impulse_current_2_4_8_16.png` |
| Schedule A | 1, 5, 7, 17 | 61x61 | **87.3%** | `gridding_impulse_schedule_A_1_5_7_17.png` |
| Schedule B | 1, 4, 9, 16 | 61x61 | **81.3%** | `gridding_impulse_schedule_B_1_4_9_16.png` |

- **Gap (A − current) = 61.5 percentage points** — far past the ~15-20pp threshold the
  brief flags as structural confirmation. The current-schedule heatmap shows a textbook
  regular checkerboard/dot-lattice pattern (visually unambiguous); Schedule A's is
  essentially solid fill (thin uncovered stripes only at the far crop edges, a boundary
  effect of the crop window, not a gap in the schedule itself).
- **Gap (B vs. A) = 6.0pp**, **(B vs. current) = 55.5pp** — Schedule B tracks much closer
  to A despite its one shared-factor pair (4 and 16). This says the *consecutive*-layer
  common factor (every adjacent pair in 2→4→8→16 shares a factor of 2) is what drives the
  damage — a single non-adjacent shared factor (B's 4↔16, two layers apart) barely
  matters by comparison.

**Task 1 verdict: gridding is structurally real and severe for the current schedule.**

## 2. Task 2 — empirical (trained-model error) periodicity

Ran on the actual test-split predictions `collect_results.py` already generated (300
held-out cases, `labelsTs` vs. `labelsPr_..._5_1_dscnoprojection_dense_dilation` — the
same split every Dice number in `results.csv` is computed from; all images are uniformly
512x512, so no resampling was needed for pixel-aligned aggregation).

Expected grid periods (rate × 8x downsample): **16px, 32px, 64px, 128px** in input-image
space.

| Map | Rate | Period | Horizontal ratio | Vertical ratio |
|---|---|---|---|---|
| FN | 2 | 16px | 0.96x | 1.56x |
| FN | 4 | 32px | 0.22x | 0.36x |
| FN | 8 | 64px | 1.43x | **2.61x** |
| FN | 16 | 128px | 0.51x | 0.31x |
| FP | 2 | 16px | 1.26x | 1.68x |
| FP | 4 | 32px | 0.73x | **2.05x** |
| FP | 8 | 64px | 1.58x | 0.99x |
| FP | 16 | 128px | 1.59x | 0.52x |

Only 2 of 16 checks (FN rate=8 vertical, FP rate=4 vertical) clear the >2x "clear peak"
bar, and neither is corroborated by its own horizontal counterpart or by neighboring
rates — inconsistent with a genuine grid (a real periodic artifact from this schedule
should show up at *all four* rates simultaneously, since all four are part of the same
repeating pattern, not appear at one isolated rate/direction). The two aggregate heatmaps
(`fn_aggregate.png`, `fp_aggregate.png`) are dominated by the vessel-tree anatomical
prior (errors cluster where vessels typically are, not a uniform noise field), and both
FFT spectra (`fn_fft_spectrum.png`, `fp_fft_spectrum.png`) show smooth, isotropic falloff
with no visible discrete rings or dots at any radius — just the expected central DC blob
and an axis-aligned cross artifact from the image-boundary discontinuity (not vessel- or
gridding-related).

**Task 2 verdict: no detectable periodic signature in real errors at the predicted grid
frequencies.** The peak/background numbers are consistent with noise, and the spectrum
images corroborate that reading visually.

## 3. Joint verdict: **inconclusive** (structural risk confirmed, not empirically dominant)

The two tasks point in different directions, and both are trustworthy on their own terms:

- Task 1 proves the current (2,4,8,16) schedule has a **severe, unambiguous structural
  gridding vulnerability** — no learned weights involved, this is pure geometry. A
  network with this dilation schedule genuinely *cannot* see ~74% of its nominal
  receptive field from any single dilated-conv stack.
- Task 2 shows this vulnerability **does not translate into a detectable periodic
  signature in the trained model's actual test-set errors.**

Plausible reasons the two disagree (not verified here, listed for the next investigation
if this gets picked up again):
- Task 1 deliberately strips BatchNorm/activations/residuals to isolate pure spatial
  reachability; the real network has all three, plus a skip connection from stage1/regular4
  and stage5, which could supply information through a completely different, non-dilated
  path that backfills what stage2/3 alone misses.
- The real model repeats the (2,4,8,16) cycle **twice** (8 blocks, not the 4 Task 1
  tested) — the second pass starts from a different post-conv feature map than the first,
  so its own grid may not be phase-aligned with the first pass's, partially self-healing
  the coverage gap that a single 4-layer stack shows in isolation.
- Real FN/FP error is dominated by broad anatomical structure (visible directly in the
  aggregate heatmaps), which may simply swamp a subtler periodic component in the FFT even
  if one exists at lower amplitude.

**This is not "gridding confirmed, go fix it" and not "gridding disproven, ignore it" —
it's a real structural weakness that isn't currently showing up as the dominant driver of
this model's specific failure modes (missed thin vessels / FP hallucination / branch
errors), at least not in a way this FFT-based test can detect.**

## 4. Recommendation

Given Schedule A (1,5,7,17) fixes a *proven, severe* structural vulnerability at
**exactly zero added params/MACs/BRAM** (same 4 layers, same kernel size, same RF, just
different dilation rate constants) — this is as close to a free experiment as exists in
this sweep. I'd frame it as a **worthwhile, low-cost robustness probe**, not a confirmed
fix for the three reported failure modes specifically:

- Retrain `S5-DscNoProjDense` with `context_pattern`'s dilation rates swapped to
  Schedule A (needs a new `DENSE_DILATION_PATTERN`-style pattern constant in `ENet.py`,
  same mechanism as the existing dense-dilation work).
- If Dice / thin-vessel recall improves, that's real evidence the structural risk *was*
  mattering despite Task 2's null result (FFT periodicity isn't the only way gridding can
  manifest — it could show up as diffuse recall loss without a clean spatial period, e.g.
  if the two repeated (2,4,8,16) cycles' grids don't stay perfectly phase-locked across
  1000+ training images with real, non-impulse content).
- If it doesn't move the needle, that corroborates Task 2 and rules gridding out as this
  model's binding constraint — worth knowing either way, at zero marginal hardware cost.
