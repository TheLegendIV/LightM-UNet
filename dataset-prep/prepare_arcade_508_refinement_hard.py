#!/usr/bin/env python
"""Build Dataset508_ARCADE_refinement_hard: a *harder*, curriculum-focused
version of Dataset507's refinement-network training data.

Dataset507 was 25% discont / 50% normal (well-predicted, unbroken) / 25%
empty_fp -- three-quarters of it was patches where the input predicted-mask
channel was already correct or already empty, which meant a network could
reach a high pooled dice by mostly copying that channel through, without
learning much about the 25% of patches (discont) this whole exercise exists
for. This dataset removes that shortcut: every non-empty patch has a
*broken* predicted mask, real or synthetic, so there is no "already right,
just copy it" case to lean on outside the empty quarter.

Three categories, 50% discont / 25% corrupt / 25% empty (parameterizable):

  discont  -- centered on real FN (residual) connected components, same
              detection as Dataset507's discontinuity_boxes. ENetOriginal's
              actual observed failures are the scarcest resource here, so
              each qualifying component contributes CROPS_PER_DISCONT
              jittered crops (not just one) to reach volume without
              inventing breaks that don't exist.
  corrupt  -- well-predicted GT-vessel patches (skeletonize + FPS coverage,
              filtered to low FN fraction -- same as Dataset507's "normal"),
              but with a synthetic break cut into the *probability* channel
              before saving: pick a bifurcation-free skeleton segment inside
              the patch (same segment-selection geometry as
              VesselGapTransform, re-implemented here without its
              batchgenerators/training-time dependencies) and suppress
              probability there by --corrupt-drop. GT and the raw image are
              untouched -- only channel 1 (the thing being refined) is
              broken. Supplements the finite real-discontinuity pool.
  empty    -- random GT-empty or FP-dominated locations, uncorrupted
              (teaches "don't add vessel that isn't there" from the model's
              real behavior, same as Dataset507's empty_or_fp).

Channel 1 is a *continuous* probability map now, not a binarized mask:
1 - probabilities[background] from nnUNetTrainerENetOriginal's saved
--save_probabilities .npz output (see run_enetoriginal_probabilities.job --
must be run first, into Dataset501_ARCADE/labelsPr_ENetOriginal_Tr and
labelsPr_ENetOriginal), scaled to uint8 [0, 255] for the same PNG pipeline
Dataset507 used. A real discontinuity is usually activation that dipped
*below* 0.5, not activation that hit zero -- binarizing throws that signal
away, and the "corrupt" category's synthetic breaks suppress probability
continuously (not hard-zero) to match.

Patches are 96x96, not 85x85 -- Dataset507's raw crops were 85x85, but
nnU-Net's planner picks patch_size independently of the (small, shallow)
custom trainer network actually used, based on its own stock deep-encoder
pooling-depth assumptions; it landed on 96x96 for that dataset, and
training silently zero-padded 85x85 up to it on every batch. Building at
96x96 natively here sidesteps that mismatch (and SmallRefinementENet's
down/up round-trip requires even H/W outright -- see ENet.py's
UpsamplingBottleneck).

Each case gets THREE files, same convention as Dataset507:
{case}_0000.png (raw grayscale), {case}_0001.png (vessel probability,
uint8), and {case}.png in labelsTr/labelsTs (GT). Case IDs:
{split}_{stem}_{category}_{idx:03d}, e.g. "train_183_discont_002".
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_arcade_vessel_coverage import farthest_point_sample

REPO_ROOT = Path(__file__).resolve().parents[1]
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"
SOURCE_NAME = "Dataset501_ARCADE"
SOURCE_DIR = NNUNET_RAW / SOURCE_NAME

PATCH_SIZE = 96
MIN_DISCONTINUITY_PX = 15    # ignore single/few-pixel FN noise, not real discontinuities
MAX_DISCONT_PER_IMAGE = 8    # cap so one heavily-mispredicted image can't dominate the pool
CROPS_PER_DISCONT = 2        # jittered crops per qualifying FN component (oversamples the finite real pool)
DISCONT_JITTER_PX = PATCH_SIZE // 6

# corrupt/empty need enough raw candidates per image that their pools can
# reach target ratio once discont sets the anchor -- same sizing Dataset507
# converged on for its "normal"/"empty_fp" pools.
CORRUPT_CANDIDATES_PER_IMAGE = 16
EMPTY_FP_ATTEMPTS_PER_IMAGE = 20

CORRUPT_DROP_FRAC = 0.9      # probability suppression fraction inside the cut box
CORRUPT_MARGIN = 4           # min skeleton pixels kept on each side of the cut, within the patch
CORRUPT_MIN_SEGMENT_PX = 15  # shortest cuttable skeleton segment inside a 96x96 patch


def load_binary(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def load_vessel_probability(npz_path: Path) -> np.ndarray:
    """1 - P(background) from nnUNetv2_predict --save_probabilities output.
    Returns float32 in [0, 1], shape (H, W).

    nnU-Net's export path keeps a leading singleton pseudo-z axis on the
    saved probabilities array even for "2d"-configured datasets (shape
    (num_classes, 1, H, W), not (num_classes, H, W)) -- squeeze it out."""
    probs = np.load(npz_path)["probabilities"]
    vessel_prob = np.squeeze(1.0 - probs[0])
    if vessel_prob.ndim != 2:
        raise ValueError(
            f"{npz_path}: expected a 2D vessel-probability map after squeezing, got shape "
            f"{vessel_prob.shape} from probabilities array shape {probs.shape}."
        )
    return vessel_prob.astype(np.float32)


def clamp_box(y: int, x: int, patch_size: int, height: int, width: int) -> Tuple[int, int, int, int]:
    half = patch_size // 2
    x0 = int(np.clip(x - half, 0, width - patch_size))
    y0 = int(np.clip(y - half, 0, height - patch_size))
    return y0, x0, y0 + patch_size, x0 + patch_size


def discontinuity_boxes_oversampled(
    gt: np.ndarray, pred: np.ndarray, patch_size: int, rng: random.Random,
) -> List[Tuple[int, int, int, int]]:
    """CROPS_PER_DISCONT jittered patches per large-enough FN (residual)
    connected component -- the first crop is always centered exactly on the
    component, the rest are jittered by up to DISCONT_JITTER_PX so they
    aren't pixel-identical while still keeping the break inside the patch."""
    height, width = gt.shape
    fn = gt & ~pred
    labeled, n = ndi.label(fn, structure=np.ones((3, 3)))
    if n == 0:
        return []
    sizes = ndi.sum(fn, labeled, range(1, n + 1))
    coms = ndi.center_of_mass(fn, labeled, range(1, n + 1))
    candidates = [(size, com) for size, com in zip(sizes, coms) if size >= MIN_DISCONTINUITY_PX]
    candidates.sort(key=lambda c: -c[0])  # largest discontinuities first
    boxes = []
    for _, (cy, cx) in candidates[:MAX_DISCONT_PER_IMAGE]:
        cy, cx = int(round(cy)), int(round(cx))
        boxes.append(clamp_box(cy, cx, patch_size, height, width))
        for _ in range(CROPS_PER_DISCONT - 1):
            jy = cy + rng.randint(-DISCONT_JITTER_PX, DISCONT_JITTER_PX)
            jx = cx + rng.randint(-DISCONT_JITTER_PX, DISCONT_JITTER_PX)
            boxes.append(clamp_box(jy, jx, patch_size, height, width))
    return boxes


def corrupt_candidate_boxes(
    gt: np.ndarray, pred: np.ndarray, patch_size: int, count: int, max_fn_frac: float = 0.1,
) -> List[Tuple[int, int, int, int]]:
    """Skeletonize + farthest-point-sample coverage of the GT mask, filtered
    to points where the prediction already matches GT well (same filter
    Dataset507's normal_boxes used) -- these are the well-predicted patches
    that get a synthetic break cut into their probability channel."""
    height, width = gt.shape
    skel = skeletonize(gt)
    points = np.argwhere(skel)
    if len(points) == 0:
        return []
    candidates = farthest_point_sample(points, min(len(points), count * 4))

    boxes = []
    for y, x in candidates:
        box = clamp_box(int(y), int(x), patch_size, height, width)
        y0, x0, y1, x1 = box
        gt_patch = gt[y0:y1, x0:x1]
        if not gt_patch.any():
            continue
        fn_patch = gt_patch & ~pred[y0:y1, x0:x1]
        if fn_patch.sum() / gt_patch.sum() <= max_fn_frac:
            boxes.append(box)
        if len(boxes) >= count:
            break
    return boxes


def empty_or_fp_boxes(
    gt: np.ndarray, pred: np.ndarray, patch_size: int, attempts: int, rng: random.Random,
    fp_frac_threshold: float = 0.05,
) -> List[Tuple[int, int, int, int]]:
    """Random locations that are either GT-empty or FP-dominated within the patch."""
    height, width = gt.shape
    if height < patch_size or width < patch_size:
        return []
    boxes = []
    fp = ~gt & pred
    for _ in range(attempts):
        y0 = rng.randint(0, height - patch_size)
        x0 = rng.randint(0, width - patch_size)
        y1, x1 = y0 + patch_size, x0 + patch_size
        gt_patch = gt[y0:y1, x0:x1]
        fp_patch = fp[y0:y1, x0:x1]
        is_empty = not gt_patch.any()
        is_fp_dominated = fp_patch.mean() >= fp_frac_threshold
        if is_empty or is_fp_dominated:
            boxes.append((y0, x0, y1, x1))
    return boxes


# -- Skeleton-segment corruption geometry --------------------------------------
# Re-implemented from vessel_gap_transform.py's _skeleton_degree/_order_path/
# _extract_segments/_pick_cut_box -- same geometry, but operating on a plain
# numpy probability patch at dataset-build time instead of a batchgenerators
# AbstractTransform on live training batches, so it doesn't need that
# module's training-loop-only dependencies.

def _skeleton_degree(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = ndi.convolve(skel.astype(int), kernel, mode="constant") - skel.astype(int)
    return neighbor_count * skel


def _order_path(component_mask: np.ndarray) -> List[Tuple[int, int]]:
    coords = set(map(tuple, np.argwhere(component_mask)))
    deg = {
        p: sum((p[0] + dy, p[1] + dx) in coords for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
        for p in coords
    }
    endpoints = [p for p, d in deg.items() if d == 1]
    start = endpoints[0] if endpoints else next(iter(coords))
    path, visited, cur = [start], {start}, start
    while len(path) < len(coords):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (cur[0] + dy, cur[1] + dx)
                if nb in coords and nb not in visited:
                    nxt = nb
                    break
            if nxt is not None:
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


def _extract_segments(skel: np.ndarray, min_length: int) -> List[List[Tuple[int, int]]]:
    deg = _skeleton_degree(skel)
    trimmed = skel & (deg < 3)
    labeled, n = ndi.label(trimmed, structure=np.ones((3, 3)))
    segments = []
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_length:
            segments.append(_order_path(comp))
    return segments


def pick_corrupt_box(mask: np.ndarray, rng: random.Random) -> Optional[Tuple[int, int, int, int]]:
    """Local (within-patch) box to suppress, centered on a bifurcation-free
    skeleton segment -- None if the patch has no cuttable segment. Returned
    as (y0, x0, y1, x1), matching clamp_box's convention (NOT
    VesselGapTransform._pick_cut_box's (y0, y1, x0, x1) -- this is a
    from-scratch return, not reused across the two modules)."""
    if mask.sum() < CORRUPT_MIN_SEGMENT_PX:
        return None
    skel = skeletonize(mask)
    segments = _extract_segments(skel, CORRUPT_MIN_SEGMENT_PX)
    if not segments:
        return None
    segment = segments[rng.randrange(len(segments))]
    margin = CORRUPT_MARGIN
    if len(segment) < 2 * margin + 2:
        center = segment[len(segment) // 2]
    else:
        center = segment[rng.randrange(margin, len(segment) - margin)]

    radius = ndi.distance_transform_edt(mask)[center]
    half_width = max(3, int(round(radius)) + 2)
    y, x = center
    h, w = mask.shape
    y0, y1 = max(0, y - half_width), min(h, y + half_width + 1)
    x0, x1 = max(0, x - half_width), min(w, x + half_width + 1)
    return y0, x0, y1, x1


def apply_probability_corruption(prob: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = box
    out = prob.copy()
    out[y0:y1, x0:x1] *= (1 - CORRUPT_DROP_FRAC)
    return out


def prob_to_u8(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob * 255.0, 0, 255).astype(np.uint8)


def save_case(
    raw: np.ndarray, prob_u8: np.ndarray, gt_mask: np.ndarray, box: Tuple[int, int, int, int],
    images_out: Path, labels_out: Path, case_id: str,
) -> None:
    y0, x0, y1, x1 = box
    Image.fromarray(raw[y0:y1, x0:x1], mode="L").save(images_out / f"{case_id}_0000.png")
    Image.fromarray(prob_u8[y0:y1, x0:x1], mode="L").save(images_out / f"{case_id}_0001.png")
    Image.fromarray(gt_mask[y0:y1, x0:x1].astype(np.uint8), mode="L").save(labels_out / f"{case_id}.png")


def process_case(
    stem: str, images_src: Path, gt_src: Path, pred_src: Path,
    images_out: Path, labels_out: Path, patch_size: int, rng: random.Random,
) -> Dict[str, List[str]]:
    raw = load_gray(images_src / f"{stem}_0000.png")
    gt = load_binary(gt_src / f"{stem}.png")
    prob = load_vessel_probability(pred_src / f"{stem}.npz")
    pred_binary = prob > 0.5

    case_ids: Dict[str, List[str]] = {"discont": [], "corrupt": [], "empty": []}

    prob_u8 = prob_to_u8(prob)
    for i, box in enumerate(discontinuity_boxes_oversampled(gt, pred_binary, patch_size, rng)):
        case_id = f"{stem}_discont_{i:03d}"
        save_case(raw, prob_u8, gt, box, images_out, labels_out, case_id)
        case_ids["discont"].append(case_id)

    for i, box in enumerate(corrupt_candidate_boxes(gt, pred_binary, patch_size, CORRUPT_CANDIDATES_PER_IMAGE)):
        y0, x0, y1, x1 = box
        gt_patch = gt[y0:y1, x0:x1]
        local_box = pick_corrupt_box(gt_patch, rng)
        if local_box is None:
            continue
        ly0, lx0, ly1, lx1 = local_box
        global_box = (y0 + ly0, x0 + lx0, y0 + ly1, x0 + lx1)
        corrupted_prob_u8 = prob_to_u8(apply_probability_corruption(prob, global_box))
        case_id = f"{stem}_corrupt_{i:03d}"
        save_case(raw, corrupted_prob_u8, gt, box, images_out, labels_out, case_id)
        case_ids["corrupt"].append(case_id)

    for i, box in enumerate(empty_or_fp_boxes(gt, pred_binary, patch_size, EMPTY_FP_ATTEMPTS_PER_IMAGE, rng)):
        case_id = f"{stem}_empty_{i:03d}"
        save_case(raw, prob_u8, gt, box, images_out, labels_out, case_id)
        case_ids["empty"].append(case_id)

    return case_ids


def _stems_from_images_dir(images_dir: Path) -> List[str]:
    return sorted(p.name[: -len("_0000.png")] for p in images_dir.glob("*_0000.png"))


def _process_split_pool(
    stems: List[str], images_src: Path, gt_src: Path, pred_src: Path,
    images_out: Path, labels_out: Path, patch_size: int, rng: random.Random,
) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {"discont": [], "corrupt": [], "empty": []}
    for stem in stems:
        result = process_case(stem, images_src, gt_src, pred_src, images_out, labels_out, patch_size, rng)
        for k in pool:
            pool[k].extend(result[k])
    return pool


def _subsample_and_prune(
    category_case_ids: List[str], target_count: int, images_out: Path, labels_out: Path, rng: random.Random,
) -> List[str]:
    if len(category_case_ids) <= target_count:
        return category_case_ids
    kept = set(rng.sample(category_case_ids, target_count))
    for case_id in category_case_ids:
        if case_id in kept:
            continue
        (images_out / f"{case_id}_0000.png").unlink(missing_ok=True)
        (images_out / f"{case_id}_0001.png").unlink(missing_ok=True)
        (labels_out / f"{case_id}.png").unlink(missing_ok=True)
    return [c for c in category_case_ids if c in kept]


def build_dataset(
    dataset_id: int,
    patch_size: int = PATCH_SIZE,
    empty_frac: float = 0.25,
    corrupt_frac: float = 0.25,
    discont_frac: float = 0.50,
    seed: int = 0,
    dataset_name: Optional[str] = None,
) -> dict:
    fracs_sum = empty_frac + corrupt_frac + discont_frac
    if abs(fracs_sum - 1.0) > 1e-6:
        raise ValueError(f"empty_frac + corrupt_frac + discont_frac must sum to 1.0, got {fracs_sum}")

    name = dataset_name or f"Dataset{dataset_id:03d}_ARCADE_refinement_hard"
    out_root = NNUNET_RAW / name
    images_tr, labels_tr = out_root / "imagesTr", out_root / "labelsTr"
    images_ts, labels_ts = out_root / "imagesTs", out_root / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    splits = json.loads((SOURCE_DIR / "splits_final.json").read_text())
    train_stems, val_stems = splits[0]["train"], splits[0]["val"]
    test_stems = _stems_from_images_dir(SOURCE_DIR / "imagesTs")

    tr_pred_dir = SOURCE_DIR / "labelsPr_ENetOriginal_Tr"
    ts_pred_dir = SOURCE_DIR / "labelsPr_ENetOriginal"
    if not (tr_pred_dir / f"{train_stems[0]}.npz").exists():
        raise FileNotFoundError(
            f"{tr_pred_dir} has no .npz probability files -- run "
            "run_enetoriginal_probabilities.job first (needs --save_probabilities, "
            "the original labelsPr_ENetOriginal_Tr/labelsPr_ENetOriginal only have binarized labels)."
        )
    if not (ts_pred_dir / f"{test_stems[0]}.npz").exists():
        raise FileNotFoundError(f"{ts_pred_dir} has no .npz probability files -- see above.")

    print(f"Building {name}: extracting all candidate patches from {len(train_stems)} train + "
          f"{len(val_stems)} val + {len(test_stems)} test source images...")

    trval_pool = _process_split_pool(
        train_stems + val_stems, SOURCE_DIR / "imagesTr", SOURCE_DIR / "labelsTr", tr_pred_dir,
        images_tr, labels_tr, patch_size, rng,
    )
    test_pool = _process_split_pool(
        test_stems, SOURCE_DIR / "imagesTs", SOURCE_DIR / "labelsTs", ts_pred_dir,
        images_ts, labels_ts, patch_size, rng,
    )

    def balance(pool: Dict[str, List[str]], images_out: Path, labels_out: Path) -> List[str]:
        n_discont = len(pool["discont"])
        if n_discont == 0:
            print("  WARNING: 0 discontinuity patches found -- discont_frac target can't be met, skipping balancing.")
            return pool["discont"] + pool["corrupt"] + pool["empty"]
        total = round(n_discont / discont_frac)
        target_corrupt = round(total * corrupt_frac)
        target_empty = round(total * empty_frac)
        kept_discont = _subsample_and_prune(pool["discont"], round(total * discont_frac), images_out, labels_out, rng)
        kept_corrupt = _subsample_and_prune(pool["corrupt"], target_corrupt, images_out, labels_out, rng)
        kept_empty = _subsample_and_prune(pool["empty"], target_empty, images_out, labels_out, rng)
        print(f"  discont={len(kept_discont)}/{n_discont} corrupt={len(kept_corrupt)}/{len(pool['corrupt'])} "
              f"empty={len(kept_empty)}/{len(pool['empty'])} (targets were "
              f"{round(total * discont_frac)}/{target_corrupt}/{target_empty})")
        return kept_discont + kept_corrupt + kept_empty

    print("Balancing train+val pool:")
    trval_cases = balance(trval_pool, images_tr, labels_tr)
    print("Balancing test pool:")
    test_cases = balance(test_pool, images_ts, labels_ts)

    train_stem_set, val_stem_set = set(train_stems), set(val_stems)

    def source_stem_of(case_id: str) -> str:
        return case_id.rsplit("_", 2)[0]

    final_train = [c for c in trval_cases if source_stem_of(c) in train_stem_set]
    final_val = [c for c in trval_cases if source_stem_of(c) in val_stem_set]
    assert len(final_train) + len(final_val) == len(trval_cases), "every trval case must map back to train or val"

    dataset_json = {
        "channel_names": {"0": "grayscale", "1": "predicted_vessel_probability"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(final_train) + len(final_val),
        "file_ending": ".png",
        "name": name,
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    splits_final = [{"train": sorted(final_train), "val": sorted(final_val)}]
    (out_root / "splits_final.json").write_text(json.dumps(splits_final, indent=2))

    return {
        "out_root": out_root, "train": final_train, "val": final_val, "test": test_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=508)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--empty-frac", type=float, default=0.25)
    parser.add_argument("--corrupt-frac", type=float, default=0.25)
    parser.add_argument("--discont-frac", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="Override the generated dataset folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(
        dataset_id=args.dataset_id, patch_size=args.patch_size,
        empty_frac=args.empty_frac, corrupt_frac=args.corrupt_frac, discont_frac=args.discont_frac,
        seed=args.seed, dataset_name=args.name,
    )
    print(f"Wrote dataset to {result['out_root']}")
    print(f"train={len(result['train'])} val={len(result['val'])} test={len(result['test'])}")


if __name__ == "__main__":
    main()
