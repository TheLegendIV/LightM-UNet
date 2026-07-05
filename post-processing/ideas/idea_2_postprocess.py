"""ARCHIVED (idea_2) -- Sobel-edge-bounded region growing via a small sliding
window repeated over a few passes.

Measured on the full 300-case ENetE1 set: mean dice 0.7467 -> 0.7453 (delta
-0.0013), 116 cases improved / 167 worsened / 17 unchanged. On the 8-case
tuning sample used to pick defaults it looked positive (+0.0022) -- that
sample was not representative of the full set.

First version (before the max_cell_size cap below existed) was much worse:
-0.10 to -0.37 mean dice, because low-gradient background regularly forms
one connected Sobel "cell" spanning tens of thousands of pixels (measured up
to ~75k px, 29% of a 512x512 frame), and any anchor touching it licensed
runaway growth into the whole thing. The cap fixed that specific failure but
the window+passes growth itself still nets slightly negative on the full set.

Superseded by idea_3 (current postprocess.py): same Sobel-cell gating, but
growth is a 4-directional whole-image raster sweep (matching idea_1's
row/column forward+backward sweep style) instead of a small local window.
Measured break-even on Dice (delta +0.0000) with a better improved/worsened
balance (132/150 vs. this file's 116/167) and higher precision on newly
added pixels (~51% vs. ~44%), at the cost of ~9x runtime.

This file is a frozen copy, kept for reference -- not imported by
run_postprocessing.py or the notebook. See postprocess.py for the active
algorithm.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BACKGROUND = 0

# Sobel gradient magnitude above this is treated as "on an edge" -- these
# pixels are excluded from every cell, so growth can never cross them.
DEFAULT_EDGE_THRESHOLD = 30.0

# Sliding-window radius (Chebyshev distance): each pass, a labeled pixel can
# grow into unlabeled pixels up to this many pixels away, provided they're in
# the same cell and appearance-consistent.
DEFAULT_WINDOW_RADIUS = 2

# Max abs intensity difference (0-255 scale) still considered "same vessel"
# as the anchor pixel it grew from.
DEFAULT_TOLERANCE = 15.0

# Number of sliding-window passes. Growth compounds pass over pass (newly
# labeled pixels become anchors for the next pass), so this bounds how far a
# label can travel in total: roughly max_passes * window_radius pixels.
DEFAULT_MAX_PASSES = 5

# A "cell" bigger than this many pixels is dropped (treated as unfillable, id
# 0) instead of used for growth. See docstring above -- this cap is
# load-bearing, not just a tuning knob.
DEFAULT_MAX_CELL_SIZE = 300


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(image, axis=1)
    gy = ndi.sobel(image, axis=0)
    return np.hypot(gx, gy)


def vessel_cells(
    image: np.ndarray,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    max_cell_size: int = DEFAULT_MAX_CELL_SIZE,
) -> np.ndarray:
    """Connected components of the non-edge (low Sobel-gradient) pixels.
    Cell id 0 means "not fillable" -- either on an edge, or in a component
    bigger than max_cell_size."""
    edge_mask = _sobel_magnitude(image) > edge_threshold
    cells, _ = ndi.label(~edge_mask, structure=ndi.generate_binary_structure(2, 2))
    cells[edge_mask] = 0

    sizes = np.bincount(cells.ravel())
    too_big = sizes > max_cell_size
    too_big[0] = True
    cells = np.where(too_big[cells], 0, cells)
    return cells


def enhance_prediction(
    image: np.ndarray,
    pred: np.ndarray,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    window_radius: int = DEFAULT_WINDOW_RADIUS,
    tolerance: float = DEFAULT_TOLERANCE,
    max_passes: int = DEFAULT_MAX_PASSES,
    max_cell_size: int = DEFAULT_MAX_CELL_SIZE,
) -> np.ndarray:
    image_f = image.astype(np.float64)
    cells = vessel_cells(image_f, edge_threshold, max_cell_size)
    enhanced = pred.copy()
    height, width = pred.shape

    for _ in range(max_passes):
        anchors = np.argwhere(enhanced != BACKGROUND)
        changed = False

        for y, x in anchors:
            cell_id = cells[y, x]
            if cell_id == BACKGROUND:
                continue

            anchor_class = enhanced[y, x]
            anchor_value = image_f[y, x]

            y0, y1 = max(0, y - window_radius), min(height, y + window_radius + 1)
            x0, x1 = max(0, x - window_radius), min(width, x + window_radius + 1)

            window_cells = cells[y0:y1, x0:x1]
            window_pred = enhanced[y0:y1, x0:x1]
            window_image = image_f[y0:y1, x0:x1]

            fillable = (
                (window_pred == BACKGROUND)
                & (window_cells == cell_id)
                & (np.abs(window_image - anchor_value) <= tolerance)
            )
            if fillable.any():
                window_pred[fillable] = anchor_class
                changed = True

        if not changed:
            break

    return enhanced
