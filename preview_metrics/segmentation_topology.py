"""Diagnostics for class-mixing, fragmentation, and anatomical consistency in
ARCADE vessel-class segmentations (Background=0, LAD=1, RCA=2, LCX=3).

RCA lives on the right coronary tree and LAD/LCX on the left tree; a single
angiogram is always exclusively one or the other, so any predicted pixel of
the "other" territory is a hard anatomical error. LAD and LCX are both on the
left tree and can only change identity at a bifurcation -- a single physical
vessel segment should not carry two labels, and a branch should never revert
to a label it already left behind further downstream.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

BACKGROUND, LAD, RCA, LCX = 0, 1, 2, 3
CLASS_LABELS = ["Background", "LAD", "RCA", "LCX"]
LEFT_CLASSES = {LAD, LCX}
EPS = 1e-8

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# 1. Territory leakage: RCA predicted inside an LAD/LCX case or vice versa.
# ---------------------------------------------------------------------------

def territory_of(classes_present: set[int]) -> str:
    has_rca = RCA in classes_present
    has_left = bool(classes_present & LEFT_CLASSES)
    if has_rca and has_left:
        return "mixed"  # not expected to occur in ARCADE ground truth
    if has_rca:
        return "RCA"
    if has_left:
        return "LAD_LCX"
    return "empty"


def territory_leakage(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Fraction of predicted foreground that lies in the anatomically wrong territory."""
    gt_classes = set(np.unique(gt).tolist()) - {BACKGROUND}
    territory = territory_of(gt_classes)
    pred_fg_px = int((pred > BACKGROUND).sum())

    if territory == "RCA":
        leaked_px = int(np.isin(pred, list(LEFT_CLASSES)).sum())
    elif territory == "LAD_LCX":
        leaked_px = int((pred == RCA).sum())
    else:
        leaked_px = 0

    return {
        "territory": territory,
        "pred_fg_px": pred_fg_px,
        "leaked_px": leaked_px,
        "territory_leakage_rate": (leaked_px / pred_fg_px) if pred_fg_px else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Component purity: does one spatially-connected blob carry >1 class?
# ---------------------------------------------------------------------------

def component_purity_rows(class_map: np.ndarray, case_id: str, connectivity: int = 2) -> list[dict]:
    """One row per connected foreground component describing its class purity."""
    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, n = ndi.label(class_map > BACKGROUND, structure=structure)
    rows = []
    for comp_id in range(1, n + 1):
        comp_mask = labeled == comp_id
        area = int(comp_mask.sum())
        classes, counts = np.unique(class_map[comp_mask], return_counts=True)
        dominant_idx = int(counts.argmax())
        rows.append({
            "case_id": case_id,
            "component_id": comp_id,
            "area_px": area,
            "dominant_class": CLASS_LABELS[int(classes[dominant_idx])],
            "purity": float(counts[dominant_idx] / counts.sum()),
            "n_classes_present": int((counts > 0).sum()),
        })
    return rows


def purity_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_components": 0, "mean_purity": np.nan, "area_weighted_purity": np.nan,
            "n_mixed_components": 0, "mixed_px_fraction": 0.0,
        }
    areas = np.array([r["area_px"] for r in rows], dtype=float)
    purities = np.array([r["purity"] for r in rows], dtype=float)
    mixed = np.array([r["n_classes_present"] for r in rows]) > 1
    return {
        "n_components": len(rows),
        "mean_purity": float(purities.mean()),
        "area_weighted_purity": float((purities * areas).sum() / areas.sum()),
        "n_mixed_components": int(mixed.sum()),
        "mixed_px_fraction": float(areas[mixed].sum() / areas.sum()) if mixed.any() else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Fragmentation: island counts vs. ground truth.
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
# 4. Branch consistency: LAD/LCX label should not revert along one vessel.
# ---------------------------------------------------------------------------

def _skeleton_degree(skel: np.ndarray) -> np.ndarray:
    padded = np.pad(skel.astype(np.uint8), 1)
    deg = np.zeros(skel.shape, dtype=np.uint8)
    for dy, dx in _NEIGHBOR_OFFSETS:
        deg += padded[1 + dy: 1 + dy + skel.shape[0], 1 + dx: 1 + dx + skel.shape[1]]
    deg[~skel] = 0
    return deg


def _skeleton_adjacency(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = [tuple(p) for p in np.argwhere(skel)]
    coord_set = set(coords)
    adj = {}
    for y, x in coords:
        adj[(y, x)] = [
            (y + dy, x + dx) for dy, dx in _NEIGHBOR_OFFSETS if (y + dy, x + dx) in coord_set
        ]
    return adj


def _bfs_path(adj: dict, start: tuple, end: tuple) -> list[tuple] | None:
    if start == end:
        return [start]
    visited = {start}
    parent = {}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb in visited:
                continue
            visited.add(nb)
            parent[nb] = node
            if nb == end:
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nb)
    return None


def _run_length_classes(values: list[int]) -> list[tuple[int, int, int]]:
    runs = []
    if not values:
        return runs
    cur, start = values[0], 0
    for i in range(1, len(values)):
        if values[i] != cur:
            runs.append((cur, start, i - start))
            cur, start = values[i], i
    runs.append((cur, start, len(values) - start))
    return runs


def _smooth_short_runs(values: list[int], min_run: int, max_passes: int = 5) -> list[int]:
    """Relabel runs shorter than min_run to a neighboring class to strip pixel-level
    flicker from skeletonization/resizing before checking for real class switches."""
    values = list(values)
    for _ in range(max_passes):
        runs = _run_length_classes(values)
        if len(runs) <= 1 or all(length >= min_run for _, _, length in runs):
            break
        changed = False
        for cls, start, length in runs:
            if length >= min_run:
                continue
            left_cls = values[start - 1] if start > 0 else None
            right_idx = start + length
            right_cls = values[right_idx] if right_idx < len(values) else None
            neighbor_cls = left_cls if left_cls is not None else right_cls
            if neighbor_cls is None:
                continue
            for i in range(start, start + length):
                values[i] = neighbor_cls
            changed = True
        if not changed:
            break
    return values


def _path_class_runs(path_pixels: list[tuple[int, int]], class_map: np.ndarray, min_run: int) -> list[int]:
    values = [int(class_map[p]) for p in path_pixels]
    values = [v for v in values if v != BACKGROUND]
    if not values:
        return []
    values = _smooth_short_runs(values, min_run=min_run)
    return [cls for cls, _, _ in _run_length_classes(values)]


def branch_consistency_stats(pred: np.ndarray, min_run_px: int = 3, max_endpoints_per_component: int = 40) -> dict:
    """Check LAD/LCX label consistency along the predicted vessel tree.

    A physical vessel branch can only change identity (LAD<->LCX) at a
    bifurcation, and once it has changed it should not revert. We
    skeletonize the predicted LAD+LCX mask and, for every pair of tree
    endpoints within the same skeleton component, read the class labels
    along the connecting path. If a class shows up, is superseded by a
    different class, and then shows up again on the same path, that's an
    illegal "LAD -> LCX -> LAD"-style reversal.
    """
    empty_result = {
        "n_endpoints": 0, "n_paths": 0, "n_inconsistent_paths": 0,
        "inconsistent_path_rate": np.nan, "skipped": False,
    }
    if skeletonize is None:
        return {**empty_result, "skipped": True}

    left_mask = np.isin(pred, list(LEFT_CLASSES))
    if left_mask.sum() < 2:
        return empty_result

    skel = skeletonize(left_mask)
    if skel.sum() < 2:
        return empty_result

    deg = _skeleton_degree(skel)
    endpoints = [tuple(p) for p in np.argwhere(skel & (deg == 1))]
    if len(endpoints) < 2:
        return {**empty_result, "n_endpoints": len(endpoints)}

    adj = _skeleton_adjacency(skel)
    skel_labeled, _ = ndi.label(skel, structure=ndi.generate_binary_structure(2, 2))

    endpoints_by_component: dict[int, list[tuple[int, int]]] = {}
    for p in endpoints:
        comp = int(skel_labeled[p])
        endpoints_by_component.setdefault(comp, []).append(p)

    n_paths = 0
    n_bad = 0
    for pts in endpoints_by_component.values():
        if len(pts) < 2:
            continue
        pts = pts[:max_endpoints_per_component]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                path = _bfs_path(adj, pts[i], pts[j])
                if not path:
                    continue
                run_classes = _path_class_runs(path, pred, min_run=min_run_px)
                if len(run_classes) < 2:
                    continue
                n_paths += 1
                if len(run_classes) != len(set(run_classes)):
                    n_bad += 1

    return {
        "n_endpoints": len(endpoints),
        "n_paths": n_paths,
        "n_inconsistent_paths": n_bad,
        "inconsistent_path_rate": (n_bad / n_paths) if n_paths else np.nan,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Convenience: everything for one (gt, pred) pair.
# ---------------------------------------------------------------------------

def evaluate_case(case_id: str, gt: np.ndarray, pred: np.ndarray, min_run_px: int = 3, min_noise_area_px: int = 5) -> dict:
    row = {"case_id": case_id}
    row.update(territory_leakage(gt, pred))
    purity_rows = component_purity_rows(pred, case_id)
    row.update(purity_summary(purity_rows))
    row.update(fragmentation_stats(gt, pred, min_area_px=min_noise_area_px))
    row.update({f"branch_{k}": v for k, v in branch_consistency_stats(pred, min_run_px=min_run_px).items()})
    return row
