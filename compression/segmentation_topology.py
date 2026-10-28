"""Diagnostics for fragmentation and component-level TP/FP structure in
ARCADE binary vessel segmentations (Background=0, Vessel=1).

Dice/precision/recall (see preview_results.ipynb) score raw pixel overlap
but say nothing about whether a prediction is a single clean vessel tree or
a scatter of fragments, or whether false positives are diffuse noise vs a
few large hallucinated blobs. This module measures exactly that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

BACKGROUND, VESSEL = 0, 1
CLASS_LABELS = ["Background", "Vessel"]
EPS = 1e-8
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# 0. Shared loading: case-ID/mask conventions used by every script in this
#    folder (and by preview_results.ipynb) so they stay in sync.
# ---------------------------------------------------------------------------

def case_id_from_image(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_0000") else stem


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_class_id_mask(path: Path, binarize: bool = True) -> np.ndarray:
    """binarize=True (default -- unchanged for every existing Dataset501-scoped
    caller in this folder) collapses to {0,1} foreground-vs-background.
    binarize=False returns the raw per-pixel class-ID array (0..K) -- used by
    compression/collect_results.py's multi-class Dataset509 per-class dice,
    which needs actual class identity, not just foreground/background."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        # Defensive fallback only: this framework should write class-ID masks, not RGB masks.
        arr = arr[..., 0]
    if binarize:
        return (arr > BACKGROUND).astype(np.uint8)  # binary: anything nonzero is vessel
    return arr.astype(np.uint8)


def resize_mask_to(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if mask.shape == (target_h, target_w):
        return mask
    return np.asarray(Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST), dtype=np.uint8)


def iter_matched_cases(
    gt_dir: Path, pred_dir: Path, binarize: bool = True
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (case_id, gt, pred) for every case present in both gt_dir and
    pred_dir, with pred resized to gt's shape when they disagree.
    binarize is forwarded to load_class_id_mask -- see its docstring."""
    pred_files = {p.stem: p for p in image_files(pred_dir)} if pred_dir.exists() else {}
    for gt_path in image_files(gt_dir):
        pred_path = pred_files.get(gt_path.stem)
        if pred_path is None:
            continue
        gt = load_class_id_mask(gt_path, binarize=binarize)
        pred = resize_mask_to(load_class_id_mask(pred_path, binarize=binarize), gt.shape)
        yield gt_path.stem, gt, pred


def dice_score(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    """Binary dice/F1: 2*TP / (2*TP + FP + FN). 1.0 when both masks are empty."""
    gt_bool = gt_bool.astype(bool)
    pred_bool = pred_bool.astype(bool)
    tp = int((gt_bool & pred_bool).sum())
    if tp == 0 and not gt_bool.any() and not pred_bool.any():
        return 1.0
    fp = int((~gt_bool & pred_bool).sum())
    fn = int((gt_bool & ~pred_bool).sum())
    return (2 * tp) / (2 * tp + fp + fn + EPS)


def cldice_score(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    """Centerline Dice (Shit et al. 2021, CVPR): harmonic mean of skeleton
    precision (predicted skeleton against GT mask) and skeleton sensitivity
    (GT skeleton against predicted mask). Reuses `skeletonize` (see
    skeleton_connectivity_stats above) -- the same primitive, not a second
    implementation of skeletonization. 1.0 when both masks are empty, 0.0 if
    exactly one is. tracked_not_gated per agent_instructions_1.yaml -- ARCADE
    annotates SYNTAX-scoreable segments, so absolute connectivity values are
    soft; this is for *relative* comparison across configs, not a gate."""
    gt_bool = gt_bool.astype(bool)
    pred_bool = pred_bool.astype(bool)
    if skeletonize is None:
        return float("nan")
    if not gt_bool.any() and not pred_bool.any():
        return 1.0
    if not gt_bool.any() or not pred_bool.any():
        return 0.0
    gt_skel = skeletonize(gt_bool)
    pred_skel = skeletonize(pred_bool)
    tprec = (pred_skel & gt_bool).sum() / (pred_skel.sum() + EPS)
    tsens = (gt_skel & pred_bool).sum() / (gt_skel.sum() + EPS)
    return float((2 * tprec * tsens) / (tprec + tsens + EPS))


# ---------------------------------------------------------------------------
# 1. Component-level TP/FP structure: is each predicted blob a real vessel
#    match or a pure hallucination? Replaces the multi-class "component
#    purity" diagnostic, which is vacuous for a single foreground class.
# ---------------------------------------------------------------------------

def component_overlap_rows(gt: np.ndarray, pred: np.ndarray, case_id: str, connectivity: int = 2) -> list[dict]:
    """One row per connected predicted-foreground component describing how
    much of it overlaps the GT vessel mask."""
    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, n = ndi.label(pred > BACKGROUND, structure=structure)
    gt_fg = gt > BACKGROUND
    rows = []
    for comp_id in range(1, n + 1):
        comp_mask = labeled == comp_id
        area = int(comp_mask.sum())
        overlap_px = int((comp_mask & gt_fg).sum())
        rows.append({
            "case_id": case_id,
            "component_id": comp_id,
            "area_px": area,
            "overlap_px": overlap_px,
            "overlap_frac": float(overlap_px / area) if area else 0.0,
        })
    return rows


def component_overlap_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_pred_components": 0, "mean_component_overlap": np.nan,
            "area_weighted_component_overlap": np.nan,
            "n_pure_fp_components": 0, "pure_fp_pixel_frac": 0.0,
        }
    areas = np.array([r["area_px"] for r in rows], dtype=float)
    overlaps = np.array([r["overlap_frac"] for r in rows], dtype=float)
    pure_fp = overlaps == 0.0
    return {
        "n_pred_components": len(rows),
        "mean_component_overlap": float(overlaps.mean()),
        "area_weighted_component_overlap": float((overlaps * areas).sum() / areas.sum()),
        "n_pure_fp_components": int(pure_fp.sum()),
        "pure_fp_pixel_frac": float(areas[pure_fp].sum() / areas.sum()) if pure_fp.any() else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Fragmentation: island counts vs. ground truth.
# ---------------------------------------------------------------------------

def fragmentation_stats(gt: np.ndarray, pred: np.ndarray, min_area_px: int = 5, connectivity: int = 2) -> dict:
    structure = ndi.generate_binary_structure(2, connectivity)
    gt_labeled, gt_n = ndi.label(gt > BACKGROUND, structure=structure)
    pred_labeled, pred_n = ndi.label(pred > BACKGROUND, structure=structure)

    n_noise_islands = 0
    if pred_n:
        sizes = ndi.sum(pred > BACKGROUND, pred_labeled, index=np.arange(1, pred_n + 1))
        n_noise_islands = int((sizes < min_area_px).sum())

    if gt_n:
        component_ratio = pred_n / gt_n
    else:
        component_ratio = float("nan") if pred_n == 0 else float("inf")

    return {
        "gt_components": int(gt_n),
        "pred_components": int(pred_n),
        "pred_noise_islands": n_noise_islands,
        "component_ratio": component_ratio,
    }


# ---------------------------------------------------------------------------
# 3. Skeleton connectivity: vessel-tree complexity, computed for both GT and
#    prediction so they can be compared directly (endpoint/branch-point
#    counts, skeleton length, number of disconnected tree fragments).
# ---------------------------------------------------------------------------

def skeleton_degree(skel: np.ndarray) -> np.ndarray:
    """8-neighbor count for each skeleton pixel (0 where not skeleton)."""
    padded = np.pad(skel.astype(np.uint8), 1)
    deg = np.zeros(skel.shape, dtype=np.uint8)
    for dy, dx in _NEIGHBOR_OFFSETS:
        deg += padded[1 + dy: 1 + dy + skel.shape[0], 1 + dx: 1 + dx + skel.shape[1]]
    deg[~skel] = 0
    return deg


def skeleton_connectivity_stats(mask: np.ndarray, connectivity: int = 2) -> dict:
    """n_endpoints (degree==1), n_branch_points (degree>=3), skeleton_length_px,
    n_skeleton_components -- a compact fingerprint of vessel-tree shape/
    complexity, independent of raw pixel-area overlap."""
    empty = {
        "n_endpoints": 0, "n_branch_points": 0,
        "skeleton_length_px": 0, "n_skeleton_components": 0, "skipped": False,
    }
    if skeletonize is None:
        return {**empty, "skipped": True}
    mask = mask.astype(bool)
    if not mask.any():
        return empty
    skel = skeletonize(mask)
    if not skel.any():
        return empty
    deg = skeleton_degree(skel)
    structure = ndi.generate_binary_structure(2, connectivity)
    _, n_comp = ndi.label(skel, structure=structure)
    return {
        "n_endpoints": int((skel & (deg == 1)).sum()),
        "n_branch_points": int((skel & (deg >= 3)).sum()),
        "skeleton_length_px": int(skel.sum()),
        "n_skeleton_components": int(n_comp),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Convenience: everything for one (gt, pred) pair.
# ---------------------------------------------------------------------------

def evaluate_case(case_id: str, gt: np.ndarray, pred: np.ndarray, min_noise_area_px: int = 5) -> dict:
    row = {"case_id": case_id}
    overlap_rows = component_overlap_rows(gt, pred, case_id)
    row.update(component_overlap_summary(overlap_rows))
    row.update(fragmentation_stats(gt, pred, min_area_px=min_noise_area_px))
    row.update({f"gt_skeleton_{k}": v for k, v in skeleton_connectivity_stats(gt > BACKGROUND).items()})
    row.update({f"pred_skeleton_{k}": v for k, v in skeleton_connectivity_stats(pred > BACKGROUND).items()})
    return row
