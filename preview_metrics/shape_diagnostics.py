"""Shape diagnostics for ARCADE predictions: class purity, cleanliness, catheter-like FPs.

Complements the dice/precision/recall/boundary_f1/cldice metrics in
preview_results.ipynb with three self-consistency signals that don't require
ground truth to be meaningful (purity, cleanliness), plus one that isolates a
specific known confounder (the contrast catheter, which has no GT class of
its own and can get mislabeled as vessel):

  1. Class purity   — does a single uninterrupted vessel branch (skeleton
                       segment between bifurcations) get assigned more than
                       one class? Splitting at junctions on purpose: LAD/RCA/
                       LCX legitimately meet at the ostium, so mixing *at* a
                       branch point isn't the failure mode this measures —
                       mixing *within* one unbranched run is.
  2. Cleanliness    — per-class fragmentation (component count, largest-
                       component ratio) and skeleton "spurs" (short branches
                       ending in a free endpoint — the standard artifact from
                       ragged mask boundaries after skeletonization).
  3. Catheter-like FP shapes — false-positive components (predicted vessel,
                       GT background) scored on elongation, skeleton junction
                       count (0 = simple path, a catheter signature), width
                       uniformity (catheters are constant-diameter; vessels
                       taper), and whether they touch the image border.
                       There's no GT catheter mask, so catheter_score is a
                       tunable heuristic, not a validated label.

Usage:
    python preview_metrics/shape_diagnostics.py --net-name ENetGlobalCtxG3
    python preview_metrics/shape_diagnostics.py --net-name ENetE1 --catheter-threshold 0.6

Writes preview_metrics/{net_name}_shape_diagnostics.csv (per image) and
preview_metrics/{net_name}_shape_diagnostics_overall.csv (mean across images).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.morphology import skeletonize

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset501_ARCADE"
LABELS_TS_DIR = DATASET_DIR / "labelsTs"

CLASS_NAMES = ("Background", "LAD", "RCA", "LCX")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Loading (mirrors preview_results.ipynb's conventions)
# ---------------------------------------------------------------------------

def case_id_from_image(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_0000") else stem


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_class_id_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def resize_mask_to(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if mask.shape == (target_h, target_w):
        return mask
    return np.asarray(Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST), dtype=np.uint8)


def build_samples(prediction_dir: Path) -> list[dict]:
    gt_files = {p.stem: p for p in image_files(LABELS_TS_DIR)}
    pred_files = {p.stem: p for p in image_files(prediction_dir)}
    samples = []
    for stem, gt_path in sorted(gt_files.items()):
        if stem in pred_files:
            samples.append({"case_id": stem, "gt": gt_path, "pred": pred_files[stem]})
    return samples


# ---------------------------------------------------------------------------
# Skeleton branch decomposition (shared by purity, cleanliness, catheter analysis)
# ---------------------------------------------------------------------------

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton_degree(skel: np.ndarray) -> np.ndarray:
    """8-neighbor count for each skeleton pixel (0 where not skeleton)."""
    neighbor_count = convolve(skel.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant", cval=0)
    return neighbor_count * skel


def skeleton_branches(mask: np.ndarray, pred: np.ndarray, n_classes: int) -> pd.DataFrame:
    """Skeletonize `mask`, split at junctions (degree >= 3), and summarize each
    resulting branch: pixel count, majority-class purity, whether it ends in
    a free endpoint (degree == 1)."""
    skel = skeletonize(mask)
    if not skel.any():
        return pd.DataFrame(columns=["size", "purity", "majority_class", "is_endpoint_branch"])
    deg = skeleton_degree(skel)
    junctions = skel & (deg >= 3)
    endpoints = skel & (deg == 1)
    branch_labels = cc_label(skel & ~junctions, connectivity=2)

    rows = []
    for lbl in range(1, int(branch_labels.max()) + 1):
        coords = np.argwhere(branch_labels == lbl)
        if coords.size == 0:
            continue
        values = pred[coords[:, 0], coords[:, 1]]
        counts = np.bincount(values, minlength=n_classes)
        majority = int(counts.argmax())
        purity = float(counts[majority] / values.size)
        is_endpoint_branch = bool(endpoints[coords[:, 0], coords[:, 1]].any())
        rows.append({
            "size": int(values.size),
            "purity": purity,
            "majority_class": majority,
            "is_endpoint_branch": is_endpoint_branch,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Class purity
# ---------------------------------------------------------------------------

def purity_metrics(branches: pd.DataFrame, min_branch_len: int) -> dict:
    usable = branches[branches["size"] >= min_branch_len]
    if usable.empty:
        return {"mean_branch_purity": np.nan, "impure_branch_frac": np.nan, "n_branches": 0}
    mean_purity = float(np.average(usable["purity"], weights=usable["size"]))
    impure_frac = float((usable["purity"] < 0.95).mean())
    return {
        "mean_branch_purity": mean_purity,
        "impure_branch_frac": impure_frac,
        "n_branches": int(len(usable)),
    }


# ---------------------------------------------------------------------------
# 2. Cleanliness
# ---------------------------------------------------------------------------

def cleanliness_metrics(pred: np.ndarray, branches: pd.DataFrame, spur_len_px: int) -> dict:
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        if c == 0:
            continue
        class_mask = pred == c
        if not class_mask.any():
            out[f"n_components_{name}"] = 0
            out[f"largest_component_ratio_{name}"] = np.nan
            continue
        labeled = cc_label(class_mask, connectivity=2)
        sizes = np.bincount(labeled.ravel())[1:]  # drop background label 0
        out[f"n_components_{name}"] = int(len(sizes))
        out[f"largest_component_ratio_{name}"] = float(sizes.max() / sizes.sum())

    spurs = branches[branches["is_endpoint_branch"] & (branches["size"] < spur_len_px)]
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
    n_classes: int,
    min_branch_len: int,
    spur_len_px: int,
    min_fp_area: int,
    catheter_threshold: float,
) -> dict:
    fg_branches = skeleton_branches(pred > 0, pred, n_classes)
    row = {}
    row.update(purity_metrics(fg_branches, min_branch_len))
    row.update(cleanliness_metrics(pred, fg_branches, spur_len_px))
    fp_components = catheter_like_fp_components(gt, pred, min_fp_area)
    row.update(catheter_summary(fp_components, catheter_threshold))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Class purity / cleanliness / catheter-FP shape diagnostics.")
    parser.add_argument("--net-name", required=True, help="Matches labelsPr_<net-name> under Dataset501_ARCADE.")
    parser.add_argument("--min-branch-len", type=int, default=3, help="Ignore skeleton branches shorter than this for purity.")
    parser.add_argument("--spur-len-px", type=int, default=8, help="Endpoint branches shorter than this count as spurs.")
    parser.add_argument("--min-fp-area", type=int, default=15, help="Ignore FP components smaller than this (speckle floor).")
    parser.add_argument("--catheter-threshold", type=float, default=0.5, help="catheter_score cutoff for 'catheter-like'.")
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "preview_metrics")
    args = parser.parse_args()

    prediction_dir = DATASET_DIR / f"labelsPr_{args.net_name}"
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {prediction_dir}")

    samples = build_samples(prediction_dir)
    if not samples:
        raise FileNotFoundError(f"No matched gt/pred pairs between {LABELS_TS_DIR} and {prediction_dir}")
    print(f"Matched samples: {len(samples)}")

    rows = []
    for item in samples:
        gt = load_class_id_mask(item["gt"])
        pred = resize_mask_to(load_class_id_mask(item["pred"]), gt.shape)
        row = diagnostics_for_image(
            gt, pred, args.n_classes, args.min_branch_len, args.spur_len_px,
            args.min_fp_area, args.catheter_threshold,
        )
        row["case_id"] = item["case_id"]
        rows.append(row)

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
