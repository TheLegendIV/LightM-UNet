#!/usr/bin/env python
"""Build Dataset505_ARCADE_oversampled from ARCADE COCO annotations, from
scratch -- NOT derived from Dataset502 like Dataset503/504 are.

prepare_arcade.py tiles each source image into a fixed rows x cols grid, so
most patches (the "empty" ones downsample_empty_patches.py exists to thin
out) never touch a vessel at all. This script instead *targets* patches at
the vessel itself:

  1. Rasterize the same binary vessel mask prepare_arcade.py uses (any
     "supported" category, stenosis excluded by default).
  2. Skeletonize it (skimage) to get the vessel centerline for the whole
     image -- reuses the exact skeleton-degree/connected-component approach
     already proven out in vessel_gap_transform.py and
     dataset-prep/preview_augmentations.ipynb, just without the branchpoint
     splitting (we want every centerline pixel as a coverage candidate here,
     not individual unbranched segments).
  3. Farthest-point sampling (FPS) on the skeleton's pixel coordinates picks
     PATCHES_PER_IMAGE well-spread centers: start from one skeleton pixel,
     then repeatedly add whichever remaining pixel is farthest from every
     already-picked center. This is the "heatmap vs skeletonize" choice the
     spec left open -- skeletonize was picked because FPS on skeleton pixels
     hits the target patch count *exactly* (or the skeleton's own pixel
     count, if the vessel is too short to support 36 well-spread points) and
     "farthest first" is precisely a coarse, coverage-maximizing sliding
     window, without needing to separately hand-tune a heatmap blur radius
     and an NMS threshold to get the same effect.
  4. Each of those points becomes the center of one PATCH_SIZE x PATCH_SIZE
     patch (clamped to stay inside the image, same clamping
     patch_grid.layer2_boxes uses). Since every center is itself a vessel
     skeleton pixel, every patch contains vessel by construction -- there is
     no empty-patch downsampling step for this dataset because there are no
     empty patches to begin with.

Images with zero vessel pixels contribute zero patches (skeleton is empty),
which is correct -- there is nothing to "cover" and no vessel content to
oversample.

Only train/val are generated (from ARCADE's own train/val image split, same
as prepare_arcade.py, so patches never leak between them since the split is
at the source-image level). No test split is generated here on purpose --
evaluation should stay on the standard, non-vessel-biased grid tiling so
metrics are comparable across experiments; point evaluation notebooks at
Dataset502_ARCADE_6x6_1c's imagesTs/labelsTs, same as Dataset503/504.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco2png import build_annotations_index, build_image_index, load_coco
from prepare_arcade import build_binary_mask, default_supported_classes, to_grayscale

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCADE_ROOT = Path(__file__).resolve().parent / "ARCADE" / "data"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"

PATCH_SIZE = 85
PATCHES_PER_IMAGE = 36


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Greedily pick up to `count` rows of `points` (shape (N, 2)), each one
    the farthest (by Euclidean distance) from every already-picked point.
    Returns at most `count` points -- fewer if len(points) < count."""
    n = len(points)
    if n == 0:
        return points
    count = min(count, n)

    picked_idx = [0]
    min_dist_to_picked = np.sum((points - points[0]) ** 2, axis=1).astype(np.float64)

    for _ in range(count - 1):
        next_idx = int(np.argmax(min_dist_to_picked))
        picked_idx.append(next_idx)
        new_dist = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist_to_picked = np.minimum(min_dist_to_picked, new_dist)

    return points[picked_idx]


def vessel_patch_boxes(
    mask: np.ndarray, patch_size: int = PATCH_SIZE, count: int = PATCHES_PER_IMAGE
) -> List[Tuple[int, int, int, int]]:
    height, width = mask.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"image {width}x{height} is smaller than patch_size={patch_size}")

    skel = skeletonize(mask.astype(bool))
    points = np.argwhere(skel)  # (N, 2) rows of (y, x)
    centers = farthest_point_sample(points, count)

    half = patch_size // 2
    boxes = []
    for y, x in centers:
        x0 = int(np.clip(x - half, 0, width - patch_size))
        y0 = int(np.clip(y - half, 0, height - patch_size))
        boxes.append((x0, y0, x0 + patch_size, y0 + patch_size))
    return boxes


def save_patch(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    box: Tuple[int, int, int, int],
    images_out: Path,
    labels_out: Path,
    case_id: str,
) -> None:
    x0, y0, x1, y1 = box
    Image.fromarray(image_arr[y0:y1, x0:x1], mode="L").save(images_out / f"{case_id}_0000.png")
    Image.fromarray(mask_arr[y0:y1, x0:x1], mode="L").save(labels_out / f"{case_id}.png")


def process_split(
    split: str,
    source_dir: Path,
    images_out: Path,
    labels_out: Path,
    patch_size: int,
    patches_per_image: int,
    supported_ids: Optional[Set[int]],
) -> List[str]:
    json_path = source_dir / split / "annotations" / f"{split}.json"
    images_dir = source_dir / split / "images"

    coco = load_coco(json_path)
    image_index = build_image_index(coco.get("images", []))
    ann_index = build_annotations_index(coco.get("annotations", []))
    ids = supported_ids if supported_ids is not None else default_supported_classes(coco)

    case_ids: List[str] = []
    n_images_with_vessel = 0

    for img_info in [image_index[k] for k in sorted(image_index)]:
        file_name = img_info["file_name"]
        img_path = images_dir / file_name

        with Image.open(img_path) as im:
            gray = np.array(to_grayscale(im))

        mask = build_binary_mask(img_info, ann_index, ids)
        stem = Path(file_name).stem

        boxes = vessel_patch_boxes(mask, patch_size, patches_per_image)
        if boxes:
            n_images_with_vessel += 1
        for i, box in enumerate(boxes):
            case_id = f"{split}_{stem}_v{i:02d}"
            save_patch(gray, mask, box, images_out, labels_out, case_id)
            case_ids.append(case_id)

    print(
        f"  {split}: {len(image_index)} source images, {n_images_with_vessel} with vessel content, "
        f"{len(case_ids)} patches ({len(case_ids) / max(n_images_with_vessel, 1):.1f}/image with vessel)"
    )
    return case_ids


def build_dataset(
    dataset_id: int,
    source: str,
    patch_size: int,
    patches_per_image: int,
    supported_ids: Optional[Set[int]] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[Path, List[str], List[str]]:
    source_dir = ARCADE_ROOT / source
    name = dataset_name or f"Dataset{dataset_id:03d}_ARCADE_oversampled"
    out_root = NNUNET_RAW / name

    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    for d in (images_tr, labels_tr):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Building {name} (patch_size={patch_size}, target {patches_per_image} patches/image)...")
    train_cases = process_split("train", source_dir, images_tr, labels_tr, patch_size, patches_per_image, supported_ids)
    val_cases = process_split("val", source_dir, images_tr, labels_tr, patch_size, patches_per_image, supported_ids)

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(train_cases) + len(val_cases),
        "file_ending": ".png",
        "name": name,
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    splits_final = [{"train": sorted(train_cases), "val": sorted(val_cases)}]
    (out_root / "splits_final.json").write_text(json.dumps(splits_final, indent=2))

    return out_root, train_cases, val_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=505)
    parser.add_argument("--source", choices=["syntax", "stenosis"], default="syntax")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patches-per-image", type=int, default=PATCHES_PER_IMAGE)
    parser.add_argument(
        "--classes",
        help="Comma-separated supported category ids to collapse into the foreground mask "
        "(default: every category except 'stenosis')",
    )
    parser.add_argument("--name", help="Override the generated dataset folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supported_ids = {int(x) for x in args.classes.split(",")} if args.classes else None

    out_root, train_cases, val_cases = build_dataset(
        dataset_id=args.dataset_id,
        source=args.source,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        supported_ids=supported_ids,
        dataset_name=args.name,
    )

    print(f"Wrote dataset to {out_root}")
    print(f"train={len(train_cases)} val={len(val_cases)}")
    print(
        "Note: no imagesTs/labelsTs generated -- point evaluation at "
        "Dataset502_ARCADE_6x6_1c's imagesTs/labelsTs (standard grid tiling, not vessel-biased)."
    )


if __name__ == "__main__":
    main()
