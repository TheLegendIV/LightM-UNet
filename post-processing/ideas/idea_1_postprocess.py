"""ARCHIVED (idea_1) -- 1D scanline region-growing via 2nd-derivative
boundaries + intensity appearance-consistency.

Measured on ENetE1 (8-case sample, then confirmed structurally on more):
hurts both Dice and clDice at every tested (tolerance, max_fill_distance)
setting. Precision on newly-added pixels topped out at ~34% regardless of how
tight the parameters were pulled (tolerance down to 3, max_fill_distance down
to 2) -- well under the ~50% breakeven needed to move Dice. Tightening the
parameters shrinks the damage (mean dice delta -0.022 at tol=15/dist=25 vs.
-0.002 at tol=3/dist=2) but it never crosses positive.

Diagnosis: a per-scanline intensity delta has no orientation/shape context,
so it can't distinguish "this high-contrast pixel is the true continuation of
the vessel" from "this pixel is just similarly dark for some other reason"
(other tissue, catheter, noise). See idea_2 (Sobel-edge-bounded region
growing) for the follow-up that adds 2D shape context.

This file is a frozen copy, kept for reference -- not imported by
run_postprocessing.py or the notebook. See postprocess.py for the active
algorithm.
"""
from __future__ import annotations

import numpy as np

BACKGROUND = 0

# Max abs intensity difference (0-255 scale) still considered "same vessel"
# as the anchor pixel it grew from. Simple/absolute for this first pass --
# see README.md for why a frame-relative version would generalize better.
DEFAULT_TOLERANCE = 15.0

# Cap on how many pixels a label may extend from its anchor, matching the
# ~20-30px scan buffer this algorithm was designed around. Guards against
# runaway fill on a long, accidentally-consistent background stretch where
# no boundary happens to be detected.
DEFAULT_MAX_FILL_DISTANCE = 25


def _second_derivative_boundaries(intensity: np.ndarray) -> np.ndarray:
    """Boolean mask, True where the 2nd derivative changes sign -- an
    intensity inflection point, i.e. a likely vessel-wall crossing."""
    d1 = np.diff(intensity.astype(np.float64), prepend=intensity[0])
    d2 = np.diff(d1, prepend=d1[0])
    sign = np.sign(d2)
    boundary = np.zeros(len(intensity), dtype=bool)
    boundary[1:] = sign[1:] != sign[:-1]
    return boundary


def _sweep_fill_1d(
    intensity: np.ndarray,
    prediction: np.ndarray,
    boundaries: np.ndarray,
    tolerance: float,
    max_fill_distance: int,
    indices,
) -> np.ndarray:
    """One directional pass (indices given in walk order): extend the most
    recently seen labeled pixel's class into background pixels ahead of it,
    stopping at the first boundary, appearance break, or distance cap.
    Encountering a new labeled pixel resets the "color" being extended."""
    enhanced = prediction.copy()
    current_class = BACKGROUND
    current_value = 0.0
    steps_since_anchor = 0

    for i in indices:
        if prediction[i] != BACKGROUND:
            current_class = int(prediction[i])
            current_value = float(intensity[i])
            steps_since_anchor = 0
            continue

        if current_class == BACKGROUND:
            continue

        steps_since_anchor += 1
        if steps_since_anchor > max_fill_distance:
            current_class = BACKGROUND
            continue

        if abs(float(intensity[i]) - current_value) > tolerance:
            current_class = BACKGROUND  # appearance broke; stop extending
            continue

        enhanced[i] = current_class
        if boundaries[i]:
            current_class = BACKGROUND  # reached the vessel wall; stop past it

    return enhanced


def _merge_agreeing_fills(original: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Accept a fill only where the two candidate results agree, or where
    only one of them proposed anything at all. Where they actively disagree
    on which class a background pixel should become, leave it unlabeled --
    that disagreement means the two directions/orientations found different
    "nearest" evidence, which is exactly the ambiguous case a first-pass
    algorithm should not guess on."""
    merged = original.copy()
    fillable = original == BACKGROUND
    agree = fillable & (a == b) & (a != BACKGROUND)
    only_a = fillable & (a != BACKGROUND) & (b == BACKGROUND)
    only_b = fillable & (b != BACKGROUND) & (a == BACKGROUND)
    merged[agree] = a[agree]
    merged[only_a] = a[only_a]
    merged[only_b] = b[only_b]
    return merged


def label_1d_segment(
    intensity: np.ndarray,
    prediction: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    max_fill_distance: int = DEFAULT_MAX_FILL_DISTANCE,
) -> np.ndarray:
    """Enhance a 1D prediction buffer using its matching 1D intensity buffer."""
    boundaries = _second_derivative_boundaries(intensity)
    forward = _sweep_fill_1d(intensity, prediction, boundaries, tolerance, max_fill_distance, range(len(intensity)))
    backward = _sweep_fill_1d(intensity, prediction, boundaries, tolerance, max_fill_distance, range(len(intensity) - 1, -1, -1))
    return _merge_agreeing_fills(prediction, forward, backward)


def enhance_prediction(
    image: np.ndarray,
    pred: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    max_fill_distance: int = DEFAULT_MAX_FILL_DISTANCE,
) -> np.ndarray:
    """2D extension: run label_1d_segment along every row and every column,
    then keep a fill only where the two orientations agree."""
    image = image.astype(np.float64)

    row_result = pred.copy()
    for r in range(pred.shape[0]):
        row_result[r] = label_1d_segment(image[r], pred[r], tolerance, max_fill_distance)

    col_result = pred.copy()
    for c in range(pred.shape[1]):
        col_result[:, c] = label_1d_segment(image[:, c], pred[:, c], tolerance, max_fill_distance)

    return _merge_agreeing_fills(pred, row_result, col_result)
