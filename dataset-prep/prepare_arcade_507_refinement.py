#!/usr/bin/env python
"""Build Dataset507_ARCADE_refinement: patches for training a *second-stage
refinement* network that takes (input image, first-pass predicted mask) and
learns to correct it against GT.

Source: Dataset501_ARCADE's full 512x512 images/GT, plus
nnUNetTrainerENetOriginal's own predictions on every split
(labelsPr_ENetOriginal_Tr for train+val, labelsPr_ENetOriginal for test --
see run_enetoriginal_inference.py for how those got generated). Predictions
are binarized (any class > 0 = vessel) before anything else here, matching
the binary-vessel convention used throughout this repo's SmallENet work.

Three patch categories, sampled from three different places in the image so
each teaches the refinement net something different:

  empty_or_fp   -- random locations that are either GT-empty or dominated by
                   false positives (model hallucinated vessel that isn't
                   there). Teaches "don't add vessel that isn't in the image."
  normal        -- GT vessel patches placed via skeletonize + farthest-point-
                   sample coverage (same technique as Dataset505/506).
                   Mostly well-predicted, ordinary examples.
  discontinuity -- centered on FN (residual) regions: connected components of
                   "GT says vessel, prediction missed it" above
                   MIN_DISCONTINUITY_PX. These are ENetOriginal's actual
                   observed failures -- the exact real-world gaps the
                   refinement net needs to learn to close, as opposed to
                   VesselGapTransform's synthetic contrast-drop gaps.

Percentages are parameterizable (--empty-frac/--normal-frac/--discont-frac,
must sum to 1.0); the discontinuity pool size is the anchor since it's the
scarcest and the whole point of this dataset -- total patch budget is
derived from how many discontinuity patches are available, then the other
two categories are sampled up to their target share of that total.

Each case gets THREE files, following nnU-Net's multi-channel input
convention: {case}_0000.png (raw grayscale), {case}_0001.png (predicted
mask), and {case}.png in labelsTr/labelsTs (GT). dataset.json declares
channel_names {"0": "grayscale", "1": "predicted_mask"}. Case IDs encode
both source split and patch category:
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

PATCH_SIZE = 85
MIN_DISCONTINUITY_PX = 15   # ignore single/few-pixel FN noise, not real discontinuities
MAX_DISCONT_PER_IMAGE = 8   # cap so one heavily-mispredicted image can't dominate the pool
# normal/empty_fp need enough raw candidates per image that their pools can
# actually reach the target ratio once discont sets the anchor (a first pass
# at 6/6 undersupplied both -- normal hit 7031/15230 target, empty_fp hit
# 4291/7615 -- so these are sized with margin above what 25/50/25 needs at
# the observed ~1 discontinuity-image-equivalent yield per source image).
NORMAL_PATCHES_PER_IMAGE = 16
EMPTY_FP_ATTEMPTS_PER_IMAGE = 20


def load_binary(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def clamp_box(y: int, x: int, patch_size: int, height: int, width: int) -> Tuple[int, int, int, int]:
    half = patch_size // 2
    x0 = int(np.clip(x - half, 0, width - patch_size))
    y0 = int(np.clip(y - half, 0, height - patch_size))
    return y0, x0, y0 + patch_size, x0 + patch_size


def discontinuity_boxes(gt: np.ndarray, pred: np.ndarray, patch_size: int) -> List[Tuple[int, int, int, int]]:
    """One patch centered on each large-enough FN (residual) connected component."""
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
        boxes.append(clamp_box(int(round(cy)), int(round(cx)), patch_size, height, width))
    return boxes


def normal_boxes(
    gt: np.ndarray, pred: np.ndarray, patch_size: int, count: int, max_fn_frac: float = 0.1,
) -> List[Tuple[int, int, int, int]]:
    """Skeletonize + farthest-point-sample coverage of the GT mask -- same
    technique as prepare_arcade_vessel_coverage.py's Dataset505/506 -- but
    filtered down to points where the prediction actually matches GT well
    (FN fraction within the patch below max_fn_frac). Without this filter,
    "normal" ends up sampling the model's failures just as often as its
    successes on a badly-mispredicted image, and stops being a meaningful
    contrast against the discontinuity category."""
    height, width = gt.shape
    skel = skeletonize(gt)
    points = np.argwhere(skel)
    if len(points) == 0:
        return []
    # Oversample candidates so filtering still leaves enough to pick `count` from.
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


def save_case(
    raw: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, box: Tuple[int, int, int, int],
    images_out: Path, labels_out: Path, case_id: str,
) -> None:
    y0, x0, y1, x1 = box
    Image.fromarray(raw[y0:y1, x0:x1], mode="L").save(images_out / f"{case_id}_0000.png")
    Image.fromarray((pred_mask[y0:y1, x0:x1].astype(np.uint8) * 255), mode="L").save(images_out / f"{case_id}_0001.png")
    Image.fromarray(gt_mask[y0:y1, x0:x1].astype(np.uint8), mode="L").save(labels_out / f"{case_id}.png")


def process_case(
    stem: str, images_src: Path, gt_src: Path, pred_src: Path,
    images_out: Path, labels_out: Path, patch_size: int, rng: random.Random,
) -> Dict[str, List[str]]:
    """`stem` (e.g. "train_183" / "val_42" / "test_277") already encodes the
    source split -- Dataset501_ARCADE's own file naming, not something we
    need to add on top of."""
    raw = load_gray(images_src / f"{stem}_0000.png")
    gt = load_binary(gt_src / f"{stem}.png")
    pred = load_binary(pred_src / f"{stem}.png")
    pred_u8 = pred  # keep boolean for box logic; save_case handles the *255 conversion

    case_ids: Dict[str, List[str]] = {"discont": [], "normal": [], "empty_fp": []}

    for i, box in enumerate(discontinuity_boxes(gt, pred, patch_size)):
        case_id = f"{stem}_discont_{i:03d}"
        save_case(raw, pred_u8, gt, box, images_out, labels_out, case_id)
        case_ids["discont"].append(case_id)

    for i, box in enumerate(normal_boxes(gt, pred, patch_size, NORMAL_PATCHES_PER_IMAGE)):
        case_id = f"{stem}_normal_{i:03d}"
        save_case(raw, pred_u8, gt, box, images_out, labels_out, case_id)
        case_ids["normal"].append(case_id)

    for i, box in enumerate(empty_or_fp_boxes(gt, pred, patch_size, EMPTY_FP_ATTEMPTS_PER_IMAGE, rng)):
        case_id = f"{stem}_empty_{i:03d}"
        save_case(raw, pred_u8, gt, box, images_out, labels_out, case_id)
        case_ids["empty_fp"].append(case_id)

    return case_ids


def _stems_from_images_dir(images_dir: Path) -> List[str]:
    return sorted(p.name[: -len("_0000.png")] for p in images_dir.glob("*_0000.png"))


def _process_split_pool(
    stems: List[str], images_src: Path, gt_src: Path, pred_src: Path,
    images_out: Path, labels_out: Path, patch_size: int, rng: random.Random,
) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {"discont": [], "normal": [], "empty_fp": []}
    for stem in stems:
        result = process_case(stem, images_src, gt_src, pred_src, images_out, labels_out, patch_size, rng)
        for k in pool:
            pool[k].extend(result[k])
    return pool


def _subsample_and_prune(
    category_case_ids: List[str], target_count: int, images_out: Path, labels_out: Path, rng: random.Random,
) -> List[str]:
    """Randomly keep target_count of category_case_ids (or all of them if
    there aren't that many); delete the on-disk files for whatever gets
    dropped, since process_case wrote every candidate unconditionally."""
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
    normal_frac: float = 0.50,
    discont_frac: float = 0.25,
    seed: int = 0,
    dataset_name: Optional[str] = None,
) -> dict:
    fracs_sum = empty_frac + normal_frac + discont_frac
    if abs(fracs_sum - 1.0) > 1e-6:
        raise ValueError(f"empty_frac + normal_frac + discont_frac must sum to 1.0, got {fracs_sum}")

    name = dataset_name or f"Dataset{dataset_id:03d}_ARCADE_refinement"
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
    if not tr_pred_dir.exists():
        raise FileNotFoundError(f"{tr_pred_dir} not found -- run inference on imagesTr first.")
    if not ts_pred_dir.exists():
        raise FileNotFoundError(f"{ts_pred_dir} not found -- run inference on imagesTs first.")

    print(f"Building {name}: extracting all candidate patches from {len(train_stems)} train + "
          f"{len(val_stems)} val + {len(test_stems)} test source images...")

    # Train+val patches go to Dataset507's imagesTr; test patches to imagesTs.
    # Kept as two separate percentage-balanced pools -- test's balance is
    # independent of train+val's so a small/large test split doesn't skew
    # imagesTr's composition or vice versa.
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
            return pool["discont"] + pool["normal"] + pool["empty_fp"]
        total = round(n_discont / discont_frac)
        target_normal = round(total * normal_frac)
        target_empty = round(total * empty_frac)
        kept_normal = _subsample_and_prune(pool["normal"], target_normal, images_out, labels_out, rng)
        kept_empty = _subsample_and_prune(pool["empty_fp"], target_empty, images_out, labels_out, rng)
        print(f"  discont={n_discont} (all kept, anchor) normal={len(kept_normal)}/{len(pool['normal'])} "
              f"empty_fp={len(kept_empty)}/{len(pool['empty_fp'])} (targets were {target_normal}/{target_empty})")
        return pool["discont"] + kept_normal + kept_empty

    print("Balancing train+val pool:")
    trval_cases = balance(trval_pool, images_tr, labels_tr)
    print("Balancing test pool:")
    test_cases = balance(test_pool, images_ts, labels_ts)

    train_stem_set, val_stem_set = set(train_stems), set(val_stems)

    def source_stem_of(case_id: str) -> str:
        # "{source_stem}_{category}_{idx:03d}" -- source_stem is itself
        # "{split}_{n}", so strip the trailing "_<category>_<idx>" segment.
        return case_id.rsplit("_", 2)[0]

    final_train = [c for c in trval_cases if source_stem_of(c) in train_stem_set]
    final_val = [c for c in trval_cases if source_stem_of(c) in val_stem_set]
    assert len(final_train) + len(final_val) == len(trval_cases), "every trval case must map back to train or val"

    dataset_json = {
        "channel_names": {"0": "grayscale", "1": "predicted_mask"},
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
    parser.add_argument("--dataset-id", type=int, default=507)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--empty-frac", type=float, default=0.25)
    parser.add_argument("--normal-frac", type=float, default=0.50)
    parser.add_argument("--discont-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="Override the generated dataset folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(
        dataset_id=args.dataset_id, patch_size=args.patch_size,
        empty_frac=args.empty_frac, normal_frac=args.normal_frac, discont_frac=args.discont_frac,
        seed=args.seed, dataset_name=args.name,
    )
    print(f"Wrote dataset to {result['out_root']}")
    print(f"train={len(result['train'])} val={len(result['val'])} test={len(result['test'])}")


if __name__ == "__main__":
    main()
