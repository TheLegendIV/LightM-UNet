# post-processing/

Goal: enhance a model's predicted masks by filling unlabeled regions that lie
between labeled ones on the same vessel, using the raw frame's intensity as
the guide. This should improve Dice on the predictions we already have,
without retraining.

## Test bench

```
post-processing/
  postprocess.py             enhance_prediction(image, pred) -> enhanced_pred.
                              The *active* algorithm -- currently idea_4
                              (Hessian/Frangi-vesselness-bounded region
                              growing). See "Algorithm history" below.
  run_postprocessing.py       CLI harness: reads imagesTs + labelsPr_<model>,
                              runs postprocess.enhance_prediction per case,
                              writes labelsPr_<model>_Pp/ here.
  preview_postprocessing.ipynb  Visual preview (raw/GT/original/enhanced) +
                              mean-dice before/after check, for one model.
  ideas/                     Frozen copies of earlier algorithm attempts,
                              kept for reference with their measured results
                              documented in each file's docstring. Not
                              imported by anything -- postprocess.py is
                              always the one actually running.
  labelsPr_<model>_Pp/       Output, one folder per model that's been run.
                              Deliberately NOT written under
                              data/nnUNet_raw/Dataset501_ARCADE/ -- that
                              directory is reserved for actual model output,
                              and post-processed masks must never be mistaken
                              for a raw prediction.
  <model>_dice_before_after.csv  Per-case dice, original vs. enhanced, from
                              the notebook's "simple test."
  idea2_vs_idea3_dice.csv    Per-case dice comparison data from calibrating
                              idea_3 against idea_2 on the full 300-case set.
  idea4_vesselness_full_sweep.csv  Per-case dice + added-pixel precision from
                              calibrating idea_4's threshold/cell-cap on the
                              full 300-case set.
```

Inputs (read, never modified):
- `data/nnUNet_raw/Dataset501_ARCADE/imagesTs` -- raw frames.
- `data/nnUNet_raw/Dataset501_ARCADE/labelsPr_<model_name>` -- that model's
  predicted class-ID masks (0=Background, 1=LAD, 2=RCA, 3=LCX). No ground
  truth is used anywhere in this pipeline -- post-processing only ever sees
  what a model would actually have available at inference time.

Run it:
```bash
python post-processing/run_postprocessing.py                # all labelsPr_* models
python post-processing/run_postprocessing.py ENetE1 LMUNet   # just these
```

To measure whether post-processing actually helps: once
`labelsPr_<model>_Pp/` exists, point `analysis/`'s tooling
(`compute_topology_metrics.py`, the notebook, `aggregate_summary.py`) at it
instead of the raw `labelsPr_<model>` folder and compare dice/purity/
fragmentation/branch-consistency before vs. after. That wiring isn't done yet
-- today those scripts only look under `data/nnUNet_raw/Dataset501_ARCADE/`.

## Algorithm history

- **idea_1** (`ideas/idea_1_postprocess.py`, pseudocode below): per-scanline
  region growing, gated by 2nd-derivative sign-change boundaries and
  intensity appearance-consistency, run along rows and columns. Implemented
  and unit-tested correctly, but measured to hurt both Dice and clDice at
  every (tolerance, max_fill_distance) setting tried -- best case still
  -0.002 mean dice on an 8-case sample, -0.026 on the full 300-case ENetE1
  set. Precision on newly-added pixels never exceeded ~34% (need >~50% to
  move Dice net-positive). Diagnosis: a 1D intensity delta has no
  orientation/shape context, so it can't tell "true vessel continuation"
  apart from "some other structure that happens to be similarly dark."
- **idea_2** (`ideas/idea_2_postprocess.py`): Sobel gradient-magnitude edges
  partition the image into connected low-gradient "cells"; a labeled pixel
  grows into unlabeled pixels of its own cell via a small sliding window,
  repeated for a few passes. Adds real 2D shape context that idea_1 lacked.
  First implementation was catastrophically worse than idea_1 (mean dice
  down to -0.10 to -0.37) because background is low-gradient too -- one cell
  regularly spans tens of thousands of pixels (measured: up to ~75k px, 29%
  of a 512x512 frame), and any anchor touching it licenses runaway growth.
  Fixed with a `max_cell_size` cap (300px) that drops any cell bigger than
  that back to "unfillable." With the cap: +0.0022 mean dice on the 8-case
  tuning sample, but only -0.0013 on the full 300-case ENetE1 set (116
  improved, 167 worsened, 17 unchanged) -- the small tuning sample was
  optimistic, not representative. Net: roughly break-even, clearly better
  than idea_1, not yet a clear win.
- **idea_3** (`ideas/idea_3_postprocess.py`): same Sobel-cell gating as
  idea_2, but instead of a small local window repeated a few times, growth
  is four whole-image raster sweeps (row-major forward/backward, column-major
  forward/backward -- state carries the entire way across the image, not just
  a local radius), merged by keeping a fill only where every sweep that
  reached a pixel agrees on the class. Full 300-case set: mean dice delta
  +0.0000 (exactly break-even), a better improved/worsened split than idea_2
  (132/150 vs. 116/167), and higher precision on newly-added pixels (~51% vs.
  ~44%, the first time crossing the ~50% Dice breakeven) -- at ~9x idea_2's
  runtime (whole-image sweeps in pure Python vs. small local windows).
- **idea_4** (current `postprocess.py`): keeps idea_3's 4-sweep-plus-agreement
  growing mechanism, but swaps the region definition from Sobel edges (any
  non-edge patch, shape-agnostic) to 2D Hessian/Frangi vesselness (real
  tubular-shape detection via the Hessian eigenvalues at each pixel and a few
  Gaussian scales -- see postprocess.py's module docstring for the math).
  Vesselness cells still occasionally chain separate vessels into one
  connected component, so the same max_cell_size safety cap from idea_2/3 is
  kept, just recalibrated much larger (5000 vs. 300) since vesselness cells
  are already far more selective than "any non-edge patch." Full 300-case
  set: mean dice 0.7467 -> 0.7485 (delta **+0.0018**), 167 improved / 126
  worsened / 7 unchanged, 46.1% mean precision on newly-added pixels -- the
  first idea in this series with a genuinely positive net effect on the full
  set, not just a tuning sample.

## idea_1 pseudocode (as originally sketched)

Input 1: Input frame
Input 2: Prediction frame
Output: enhanced prediction frame

Goal: grow labelled regions to improve Dice

FUNCTION label_1d_segment(intensity[N], prediction[N]):
    # intensity[]  : greyscale values along the scan line (prefilled 20-30 px buffer)
    # prediction[] : current binary labels along the same line (0/1)
    # returns enhanced_prediction[]

    enhanced = copy(prediction)

    # --- Step 1: compute derivatives along the buffer ---
    d1 = first_derivative(intensity)     # d1[i] = intensity[i] - intensity[i-1]
    d2 = second_derivative(intensity)    # d2[i] = d1[i] - d1[i-1]

    # --- Step 2: find boundary indices (zero-crossings of d2) ---
    boundaries = []
    FOR i FROM 1 TO N-1:
        IF sign(d2[i]) != sign(d2[i-1]):   # d2 changed sign => inflection => boundary
            boundaries.append(i)

    # --- Step 3: find labeled anchor pixels ---
    labeled_indices = [i FOR i IN range(N) IF prediction[i] == 1]

    IF labeled_indices is empty:
        RETURN enhanced        # nothing to grow from on this line

    # --- Step 4: for each labeled pixel, fill toward the nearest boundary on each side ---
    FOR each L IN labeled_indices:
        vessel_value = intensity[L]        # appearance reference: the labeled pixel's intensity

        # ---- fill BACKWARD from L to the previous boundary ----
        prev_boundary = largest boundary index that is < L, else 0
        FOR i FROM L-1 DOWN TO prev_boundary:
            IF appearance_consistent(intensity[i], vessel_value):
                enhanced[i] = 1
            ELSE:
                BREAK          # appearance broke before boundary => stop

        # ---- fill FORWARD from L to the next boundary ----
        next_boundary = smallest boundary index that is > L, else N-1
        FOR i FROM L+1 UP TO next_boundary:
            IF appearance_consistent(intensity[i], vessel_value):
                enhanced[i] = 1
            ELSE:
                BREAK

    RETURN enhanced


FUNCTION appearance_consistent(pixel_value, vessel_value):
    # relative, frame-invariant check: is this pixel close enough in intensity
    # to the reference vessel pixel to be considered the same structure?
    RETURN abs(pixel_value - vessel_value) <= TOLERANCE
    # TOLERANCE is your one tuning parameter (see note below)

