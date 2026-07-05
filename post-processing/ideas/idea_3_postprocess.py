"""ARCHIVED (idea_3): Sobel-edge-bounded, 4-directional raster-sweep region
growing.

Superseded by idea_4 (current postprocess.py), which keeps this file's
4-directional-sweep-plus-agreement growing mechanism but swaps the region
definition from generic Sobel edges (any intensity edge, vessel or not) to
Hessian/Frangi vesselness (specifically tubular, vessel-shaped structure) --
testing whether real shape context beats "is this an edge at all."

This file is a frozen copy, kept for reference -- not imported by
run_postprocessing.py or the notebook. See postprocess.py for the active
algorithm.

Same Sobel-edge "cell" gating as idea_2 (ideas/idea_2_postprocess.py): the
image is partitioned into connected components of low-gradient pixels, and a
label can only grow into unlabeled pixels of its own cell. Where idea_2 grew
each anchor into a small local window repeated over a few passes, this
version grows via four whole-image raster sweeps, each carrying state the
entire way across the image rather than resetting every window_radius
pixels:

  1. row-major forward:   left -> right, row by row, top -> bottom
  2. column-major forward: top -> bottom, column by column, left -> right
  3. row-major backward:  right -> left, row by row, bottom -> top
  4. column-major backward: bottom -> top, column by column, right -> left

Each sweep walks its scan order carrying a "current class" (the most
recently seen labeled pixel, i.e. the anchor "color" to extend, matching
idea_1's rule that a new label overrides the color being grown) and fills
unlabeled pixels ahead of it as long as they're in the same cell and
appearance-consistent, resetting whenever either check fails. A pixel is
only filled in the final result if every sweep that reached it agrees on the
class -- sweeps that latched onto a distant, less-relevant anchor along
their direction tend to disagree with the others, which is what keeps this
more precise than idea_2 despite reaching much farther per pass.

Measured on the full 300-case ENetE1 set vs. idea_2: break-even mean dice
(delta +0.0000 vs. idea_2's -0.0013), a better improved/worsened balance
(132/150 vs. 116/167), and higher precision on newly added pixels (~51% vs.
~44%, crossing the ~50% breakeven for the first time) -- at ~9x the runtime
(whole-image sweeps in pure Python vs. small local windows).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BACKGROUND = 0

# Sobel gradient magnitude above this is treated as "on an edge" -- these
# pixels are excluded from every cell, so growth can never cross them.
DEFAULT_EDGE_THRESHOLD = 30.0

# Max abs intensity difference (0-255 scale) still considered "same vessel"
# as the anchor pixel currently being grown.
DEFAULT_TOLERANCE = 15.0

# A "cell" bigger than this many pixels is dropped (treated as unfillable, id
# 0) instead of used for growth. Real vessel background is low-gradient too,
# so without this cap, low-gradient background regularly forms one connected
# blob spanning tens of thousands of pixels (measured: up to ~75k px, ~29% of
# a 512x512 frame) -- any anchor that happens to touch it licenses growth
# into the whole blob. Load-bearing, not just a tuning knob.
DEFAULT_MAX_CELL_SIZE = 300

# The four (major axis, row direction, col direction) sweep orders described
# in the module docstring.
_SWEEP_ORDERS = [
    ("row", 1, 1),
    ("col", 1, 1),
    ("row", -1, -1),
    ("col", -1, -1),
]


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
    bigger than max_cell_size. A label can never grow across a 0."""
    edge_mask = _sobel_magnitude(image) > edge_threshold
    cells, _ = ndi.label(~edge_mask, structure=ndi.generate_binary_structure(2, 2))
    cells[edge_mask] = 0

    sizes = np.bincount(cells.ravel())
    too_big = sizes > max_cell_size
    too_big[0] = True  # id 0 is already "not fillable"; keep it that way
    cells = np.where(too_big[cells], 0, cells)
    return cells


def _raster_sweep_fill(
    image_f: np.ndarray,
    prediction: np.ndarray,
    cells: np.ndarray,
    tolerance: float,
    major: str,
    row_dir: int,
    col_dir: int,
) -> np.ndarray:
    """One directional whole-image sweep. `major` picks row-major (scan each
    row left-to-right or right-to-left, rows in row_dir order) or col-major
    (scan each column top-to-bottom or bottom-to-top, columns in col_dir
    order). Carries a single "current class" across the entire sweep,
    reset whenever a real label, a cell boundary, or an appearance break is
    encountered -- see module docstring."""
    enhanced = prediction.copy()
    height, width = prediction.shape
    rows = range(height)[::row_dir]
    cols = range(width)[::col_dir]
    outer, inner = (rows, cols) if major == "row" else (cols, rows)

    current_class = BACKGROUND
    current_value = 0.0
    current_cell = BACKGROUND

    for o in outer:
        for i in inner:
            y, x = (o, i) if major == "row" else (i, o)

            if prediction[y, x] != BACKGROUND:
                current_class = int(prediction[y, x])
                current_value = float(image_f[y, x])
                current_cell = cells[y, x]
                continue

            if current_class == BACKGROUND:
                continue

            if cells[y, x] != current_cell or cells[y, x] == BACKGROUND:
                current_class = BACKGROUND  # left the anchor's cell; stop extending
                continue

            if abs(float(image_f[y, x]) - current_value) > tolerance:
                current_class = BACKGROUND  # appearance broke; stop extending
                continue

            enhanced[y, x] = current_class

    return enhanced


def _merge_agreeing_fills(original: np.ndarray, sweep_results: list[np.ndarray]) -> np.ndarray:
    """Accept a fill only where every sweep that proposed something for a
    pixel agrees on the class. A pixel no sweep reached stays background;
    a pixel where sweeps disagree stays background too -- that disagreement
    means different directions latched onto different "nearest" anchors,
    exactly the ambiguous case this shouldn't guess on."""
    merged = original.copy()
    fillable = original == BACKGROUND
    any_fill = np.zeros(original.shape, dtype=bool)
    agree = np.ones(original.shape, dtype=bool)
    fill_class = np.full(original.shape, BACKGROUND, dtype=original.dtype)

    for result in sweep_results:
        proposed = result != BACKGROUND
        conflicting = proposed & (fill_class != BACKGROUND) & (fill_class != result)
        agree &= ~conflicting
        fill_class = np.where(proposed & (fill_class == BACKGROUND), result, fill_class)
        any_fill |= proposed

    do_fill = fillable & any_fill & agree
    merged[do_fill] = fill_class[do_fill]
    return merged


def enhance_prediction(
    image: np.ndarray,
    pred: np.ndarray,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
    max_cell_size: int = DEFAULT_MAX_CELL_SIZE,
) -> np.ndarray:
    """Post-processing entry point used by run_postprocessing.py.

    Args:
        image: raw grayscale frame, shape (H, W), uint8.
        pred: predicted class-ID mask, shape (H, W), uint8, values in
            {0=Background, 1=LAD, 2=RCA, 3=LCX}.

    Returns:
        Enhanced class-ID mask, same shape/dtype as pred.
    """
    image_f = image.astype(np.float64)
    cells = vessel_cells(image_f, edge_threshold, max_cell_size)

    sweep_results = [
        _raster_sweep_fill(image_f, pred, cells, tolerance, major, row_dir, col_dir)
        for major, row_dir, col_dir in _SWEEP_ORDERS
    ]
    return _merge_agreeing_fills(pred, sweep_results)
