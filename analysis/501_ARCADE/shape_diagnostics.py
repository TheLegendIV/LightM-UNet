"""Shape diagnostics for ARCADE binary vessel predictions: cleanliness,
catheter-like FPs.

Complements the dice/precision/recall/boundary_f1/cldice metrics in
preview_results.ipynb with a self-consistency signal that doesn't require
ground truth to be meaningful (cleanliness), plus one that isolates a
specific known confounder (the contrast catheter, which has no GT class of
its own and can get mislabeled as vessel):

  1. Cleanliness    — vessel-mask fragmentation (component count, largest-
                       component ratio) and skeleton "spurs" (short branches
                       ending in a free endpoint — the standard artifact from
                       ragged mask boundaries after skeletonization).
  2. Catheter-like FP shapes — false-positive components (predicted vessel,
                       GT background) scored on elongation, skeleton junction
                       count (0 = simple path, a catheter signature), width
                       uniformity (catheters are constant-diameter; vessels
                       taper), and whether they touch the image border.
                       There's no GT catheter mask, so catheter_score is a
                       tunable heuristic, not a validated label.

(Class purity -- does one skeleton branch carry more than one label -- is
not meaningful for a binary background/vessel mask, since every foreground
pixel is trivially "the same class"; that diagnostic lived here when this
folder covered 4-class LAD/RCA/LCX segmentation and has been dropped. See
segmentation_topology.py for the skeleton-connectivity stats that replaced it.)

Usage:
    python analysis/501_ARCADE/shape_diagnostics.py --net-name nnUNetTrainerENet_E1
    python analysis/501_ARCADE/shape_diagnostics.py --net-name nnUNetTrainerENet_Original --catheter-threshold 0.6

Writes analysis/501_ARCADE/results/{net_name}_shape_diagnostics.csv (per image) and
analysis/501_ARCADE/results/{net_name}_shape_diagnostics_overall.csv (mean across images).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation_topology as topo
from segmentation_topology import skeleton_degree

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset501_ARCADE"
LABELS_TS_DIR = DATASET_DIR / "labelsTs"

CLASS_NAMES = ("Background", "Vessel")


# ---------------------------------------------------------------------------
# Skeleton branch decomposition (shared by cleanliness, catheter analysis)
# ---------------------------------------------------------------------------

def skeleton_branches(mask: np.ndarray) -> pd.DataFrame:
    """Skeletonize `mask`, split at junctions (degree >= 3), and summarize each
    resulting branch: pixel count, whether it ends in a free endpoint
    (degree == 1)."""
    skel = skeletonize(mask)
    if not skel.any():
        return pd.DataFrame(columns=["size", "is_endpoint_branch"])
    deg = skeleton_degree(skel)
    junctions = skel & (deg >= 3)
    endpoints = skel & (deg == 1)
    branch_labels = cc_label(skel & ~junctions, connectivity=2)

    rows = []
    for lbl in range(1, int(branch_labels.max()) + 1):
        coords = np.argwhere(branch_labels == lbl)
        if coords.size == 0:
            continue
        is_endpoint_branch = bool(endpoints[coords[:, 0], coords[:, 1]].any())
        rows.append({
            "size": int(coords.shape[0]),
            "is_endpoint_branch": is_endpoint_branch,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Cleanliness
# ---------------------------------------------------------------------------

def cleanliness_metrics(pred: np.ndarray, branches: pd.DataFrame, spur_len_px: int) -> dict:
    out = {}
    class_mask = pred > 0
    if not class_mask.any():
        out["n_components_Vessel"] = 0
        out["largest_component_ratio_Vessel"] = np.nan
    else:
        labeled = cc_label(class_mask, connectivity=2)
        sizes = np.bincount(labeled.ravel())[1:]  # drop background label 0
        out["n_components_Vessel"] = int(len(sizes))
        out["largest_component_ratio_Vessel"] = float(sizes.max() / sizes.sum())

    spurs = branches[branches["is_endpoint_branch"] & (branches["size"] < spur_len_px)]
    out["n_skeleton_branches"] = int(len(branches))
    out["spur_count"] = int(len(spurs))
    out["spur_density"] = float(len(spurs) / len(branches)) if len(branches) else np.nan
    return out


# ---------------------------------------------------------------------------
# 3. Catheter-like false-positive shapes
# ---------------------------------------------------------------------------

def catheter_like_fp_components(gt: np.ndarray, pred: np.ndarray, min_area: int) -> pd.DataFrame:
    fp_mask = (pred > 0) & (gt == 0)
    labeled = cc_label(fp_mask, connectivity=2)
    H, W = gt.shape
    rows = []
    for p in regionprops(labeled):
        if p.area < min_area:
            continue
        minr, minc, maxr, maxc = p.bbox
        comp_mask = labeled[minr:maxr, minc:maxc] == p.label
        touches_border = minr == 0 or minc == 0 or maxr == H or maxc == W

        minor = max(p.minor_axis_length, 1e-6)
        axis_ratio = float(p.major_axis_length / minor)

        skel = skeletonize(comp_mask)
        n_junctions = int((skeleton_degree(skel) >= 3).sum())

        dist = distance_transform_edt(comp_mask)
        widths = dist[skel]
        width_cv = float(widths.std() / (widths.mean() + 1e-8)) if widths.size else np.nan

        predicted_class = int(np.bincount(pred[minr:maxr, minc:maxc][comp_mask]).argmax())

        elongation_term = min(axis_ratio / 8.0, 1.0)
        width_term = 1.0 - min(width_cv, 1.0) if np.isfinite(width_cv) else 0.5
        junction_term = 1.0 if n_junctions == 0 else 0.4
        catheter_score = float(elongation_term * width_term * junction_term)

        rows.append({
            "area": int(p.area),
            "axis_ratio": axis_ratio,
            "solidity": float(p.solidity),
            "n_junctions": n_junctions,
            "width_cv": width_cv,
            "touches_border": touches_border,
            "predicted_class": predicted_class,
            "catheter_score": catheter_score,
        })
    return pd.DataFrame(rows)


def catheter_summary(fp_components: pd.DataFrame, catheter_threshold: float) -> dict:
    if fp_components.empty:
        return {"n_fp_components": 0, "n_catheter_like_fp": 0, "catheter_like_fp_pixel_frac": np.nan}
    catheter_like = fp_components[fp_components["catheter_score"] >= catheter_threshold]
    total_fp_pixels = fp_components["area"].sum()
    return {
        "n_fp_components": int(len(fp_components)),
        "n_catheter_like_fp": int(len(catheter_like)),
        "catheter_like_fp_pixel_frac": float(catheter_like["area"].sum() / total_fp_pixels) if total_fp_pixels else np.nan,
    }


# ---------------------------------------------------------------------------
# Per-image driver + CLI
# ---------------------------------------------------------------------------

def diagnostics_for_image(
    gt: np.ndarray,
    pred: np.ndarray,
    spur_len_px: int,
    min_fp_area: int,
    catheter_threshold: float,
) -> dict:
    fg_branches = skeleton_branches(pred > 0)
    row = {}
    row.update(cleanliness_metrics(pred, fg_branches, spur_len_px))
    fp_components = catheter_like_fp_components(gt, pred, min_fp_area)
    row.update(catheter_summary(fp_components, catheter_threshold))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanliness / catheter-FP shape diagnostics for binary vessel masks.")
    parser.add_argument("--net-name", required=True, help="Matches labelsPr_<net-name> under Dataset501_ARCADE.")
    parser.add_argument("--spur-len-px", type=int, default=8, help="Endpoint branches shorter than this count as spurs.")
    parser.add_argument("--min-fp-area", type=int, default=15, help="Ignore FP components smaller than this (speckle floor).")
    parser.add_argument("--catheter-threshold", type=float, default=0.5, help="catheter_score cutoff for 'catheter-like'.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis" / "501_ARCADE" / "results")
    args = parser.parse_args()

    prediction_dir = DATASET_DIR / f"labelsPr_{args.net_name}"
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {prediction_dir}")

    rows = []
    for case_id, gt, pred in topo.iter_matched_cases(LABELS_TS_DIR, prediction_dir):
        row = diagnostics_for_image(
            gt, pred, args.spur_len_px, args.min_fp_area, args.catheter_threshold,
        )
        row["case_id"] = case_id
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No matched gt/pred pairs between {LABELS_TS_DIR} and {prediction_dir}")
    print(f"Matched samples: {len(rows)}")

    per_image = pd.DataFrame(rows)
    cols = ["case_id"] + [c for c in per_image.columns if c != "case_id"]
    per_image = per_image[cols]

    numeric_cols = [c for c in per_image.columns if c != "case_id"]
    overall = pd.DataFrame([per_image[numeric_cols].mean(numeric_only=True)])

    args.out_dir.mkdir(exist_ok=True)
    per_image_path = args.out_dir / f"{args.net_name}_shape_diagnostics.csv"
    overall_path = args.out_dir / f"{args.net_name}_shape_diagnostics_overall.csv"
    per_image.to_csv(per_image_path, index=False)
    overall.to_csv(overall_path, index=False)

    print("\n=== Overall (mean across images) ===")
    print(overall.to_string(index=False))
    print(f"\nSaved: {per_image_path}")
    print(f"Saved: {overall_path}")


if __name__ == "__main__":
    main()
