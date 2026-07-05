"""Post-processing algorithm (idea_4): Hessian/Frangi-vesselness-bounded,
4-directional raster-sweep region growing.

Keeps idea_3's growing mechanism (ideas/idea_3_postprocess.py) unchanged --
four whole-image raster sweeps, merged by agreement -- and swaps the region
("cell") definition. idea_3 used Sobel gradient magnitude: a cell was any
connected component of "not a strong edge," which includes any locally
homogeneous patch regardless of shape (flat background counts too). This
version uses 2D Hessian/Frangi vesselness instead: at each pixel and Gaussian
scale, the Hessian's eigenvalues describe local curvature -- a tubular ridge
has one near-zero eigenvalue (along the vessel, low curvature) and one large
eigenvalue of consistent sign (across the vessel, high curvature). The
classic Frangi formula turns that eigenvalue pair into a single "how much
does this look like a tube, not a blob or flat region" score. A cell is now a
connected component of specifically vessel-shaped structure, not just "any
non-edge region" -- real 2D shape context, which idea_1's 1D derivative and
idea_2/3's generic Sobel edges both lacked.

ARCADE vessels are dark ridges (contrast agent absorbs X-rays), so the image
is intensity-inverted before the Hessian is computed, matching the classic
Frangi formula's assumption of bright ridges.

Vesselness alone is a noisy detector on its own (calibration on real frames:
precision on GT vessel pixels only ~0.15-0.26 for masks thresholded on
vesselness alone, at various thresholds) -- but that's fine here, since it's
never used to invent labels from nothing. It only gates where an *existing*
labeled pixel is allowed to grow, same as idea_2/3's Sobel cells.

Low-vesselness connected components can still occasionally span thousands of
pixels (faint vessel-like noise chaining separate real vessels together),
just less catastrophically than Sobel's ~75k-px background blobs -- the same
max_cell_size cap from idea_2/3 is kept as a safety net, just recalibrated
much larger (5000 vs. 300) since vesselness cells are already far more
selective than Sobel's "any non-edge patch."

Measured on the full 300-case ENetE1 set: mean dice 0.7467 -> 0.7485 (delta
+0.0018), 167 cases improved / 126 worsened / 7 unchanged, mean precision on
newly-added pixels 46.1% -- the first idea in this series with a genuinely
positive net effect on the full set (idea_1: -0.026, idea_2: -0.0013, idea_3:
+0.0000). Chosen over a similarly-scored alternative (vesselness_threshold
0.05 with max_cell_size 20000: delta +0.0018 too, but a worse 156/142
improved/worsened split and lower 42.3% precision).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BACKGROUND = 0

# Gaussian scales (pixel radius) the Hessian is evaluated at; the final
# vesselness per pixel is the max response across scales. Should roughly
# bracket the range of vessel widths present in the image.
DEFAULT_SIGMAS = (1.0, 2.0, 3.0)

# Frangi shape parameters. beta controls sensitivity to blob-vs-line shape
# (Rb = lambda1/lambda2); c controls sensitivity to overall response
# magnitude (S = sqrt(lambda1^2+lambda2^2)), scaled for 0-255 intensities.
DEFAULT_FRANGI_BETA = 0.5
DEFAULT_FRANGI_C = 15.0

# Vesselness score (0-1) above which a pixel is considered part of a
# vessel-shaped cell.
DEFAULT_VESSELNESS_THRESHOLD = 0.08

# Max abs intensity difference (0-255 scale) still considered "same vessel"
# as the anchor pixel currently being grown.
DEFAULT_TOLERANCE = 15.0

# A "cell" bigger than this many pixels is dropped (treated as unfillable, id
# 0) instead of used for growth -- see module docstring.
DEFAULT_MAX_CELL_SIZE = 5000

# The four (major axis, row direction, col direction) sweep orders: row-major
# forward/backward and column-major forward/backward (unchanged from idea_3).
_SWEEP_ORDERS = [
    ("row", 1, 1),
    ("col", 1, 1),
    ("row", -1, -1),
    ("col", -1, -1),
]


def _hessian_eigenvalues(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale-normalized 2D Hessian eigenvalues at one Gaussian scale, sorted
    so |lambda1| <= |lambda2|."""
    hyy = ndi.gaussian_filter(image, sigma=sigma, order=(2, 0)) * sigma**2
    hxx = ndi.gaussian_filter(image, sigma=sigma, order=(0, 2)) * sigma**2
    hxy = ndi.gaussian_filter(image, sigma=sigma, order=(1, 1)) * sigma**2

    trace = hyy + hxx
    disc = np.sqrt(np.maximum((hyy - hxx) ** 2 + 4 * hxy**2, 0.0))
    mu1 = 0.5 * (trace - disc)
    mu2 = 0.5 * (trace + disc)

    swap = np.abs(mu1) > np.abs(mu2)
    lambda1 = np.where(swap, mu2, mu1)
    lambda2 = np.where(swap, mu1, mu2)
    return lambda1, lambda2


def _frangi_vesselness(lambda1: np.ndarray, lambda2: np.ndarray, beta: float, c: float) -> np.ndarray:
    """Classic 2D Frangi vesselness for bright ridges (lambda2 < 0)."""
    eps = 1e-8
    blobness = lambda1 / (lambda2 + eps)
    structureness = np.sqrt(lambda1**2 + lambda2**2)
    vesselness = np.exp(-(blobness**2) / (2 * beta**2)) * (1 - np.exp(-(structureness**2) / (2 * c**2)))
    return np.where(lambda2 > 0, 0.0, vesselness)


def vesselness_map(
    image: np.ndarray,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    beta: float = DEFAULT_FRANGI_BETA,
    c: float = DEFAULT_FRANGI_C,
) -> np.ndarray:
    """Multi-scale Frangi vesselness, one score per pixel in [0, 1]."""
    inverted = 255.0 - image.astype(np.float64)  # ARCADE vessels are dark; Frangi assumes bright ridges
    best = np.zeros(image.shape, dtype=np.float64)
    for sigma in sigmas:
        lambda1, lambda2 = _hessian_eigenvalues(inverted, sigma)
        best = np.maximum(best, _frangi_vesselness(lambda1, lambda2, beta, c))
    return best


def vessel_cells(
    image: np.ndarray,
    vesselness_threshold: float = DEFAULT_VESSELNESS_THRESHOLD,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
    max_cell_size: int = DEFAULT_MAX_CELL_SIZE,
) -> np.ndarray:
    """Connected components of vessel-shaped (high-vesselness) pixels. Cell
    id 0 means "not fillable" -- either not vessel-shaped, or in a component
    bigger than max_cell_size. A label can never grow across a 0."""
    vessel_mask = vesselness_map(image, sigmas) > vesselness_threshold
    cells, _ = ndi.label(vessel_mask, structure=ndi.generate_binary_structure(2, 2))

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
    """One directional whole-image sweep (unchanged from idea_3). Carries a
    single "current class" across the entire sweep -- the most recently seen
    labeled pixel -- and fills unlabeled pixels ahead of it as long as
    they're in the same cell and appearance-consistent, resetting on
    whichever check fails first."""
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
    pixel agrees on the class (unchanged from idea_3)."""
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
    vesselness_threshold: float = DEFAULT_VESSELNESS_THRESHOLD,
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
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
    cells = vessel_cells(image_f, vesselness_threshold, sigmas, max_cell_size)

    sweep_results = [
        _raster_sweep_fill(image_f, pred, cells, tolerance, major, row_dir, col_dir)
        for major, row_dir, col_dir in _SWEEP_ORDERS
    ]
    return _merge_agreeing_fills(pred, sweep_results)
