#!/usr/bin/env python
"""Build an nnUNet-style raw dataset from ARCADE COCO annotations.

Each source image is grayscale-converted and rasterized into a binary mask
(foreground = any "supported" category), then tiled into a rows x cols grid
of layer-1 patches. Optionally, a second (rows-1) x (cols-1) grid of
layer-2 patches is generated, each centered on an internal layer-1 grid
vertex so it straddles the surrounding layer-1 borders. Patches are written
directly as nnUNet cases (imagesTr/labelsTr for train+val, imagesTs/labelsTs
for test), named "{split}_{source_image_stem}_p{row}{col}" (layer 1) /
"..._b{row}{col}" (layer 2, "b" for border).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco2png import build_annotations_index, build_image_index, load_coco, polygon_to_mask
from patch_grid import Box, layer1_boxes, layer2_boxes

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCADE_ROOT = Path(__file__).resolve().parent / "ARCADE" / "data"
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"

SPLITS = ("train", "val", "test")


def to_grayscale(img: Image.Image) -> Image.Image:
    return img if img.mode == "L" else img.convert("L")


def default_supported_classes(coco: Dict, exclude_names: Iterable[str] = ("stenosis",)) -> Set[int]:
    exclude = set(exclude_names)
    return {c["id"] for c in coco.get("categories", []) if c.get("name") not in exclude}


def build_binary_mask(img_info: Dict, ann_index: Dict[int, List[Dict]], supported_ids: Set[int]) -> np.ndarray:
    width = int(img_info["width"])
    height = int(img_info["height"])
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in ann_index.get(img_info["id"], []):
        if int(ann.get("category_id", 0)) not in supported_ids:
            continue

        segmentation = ann.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue

        part = polygon_to_mask(segmentation, width, height, fill_value=1)
        mask = np.maximum(mask, part)

    return mask


def save_patch(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    box: Box,
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
    cols: int,
    rows: int,
    include_layer2: bool,
    supported_ids: Optional[Set[int]],
) -> List[str]:
    json_path = source_dir / split / "annotations" / f"{split}.json"
    images_dir = source_dir / split / "images"

    coco = load_coco(json_path)
    image_index = build_image_index(coco.get("images", []))
    ann_index = build_annotations_index(coco.get("annotations", []))
    ids = supported_ids if supported_ids is not None else default_supported_classes(coco)

    case_ids: List[str] = []

    for img_info in [image_index[k] for k in sorted(image_index)]:
        file_name = img_info["file_name"]
        img_path = images_dir / file_name

        with Image.open(img_path) as im:
            gray = np.array(to_grayscale(im))

        mask = build_binary_mask(img_info, ann_index, ids)
        width, height = int(img_info["width"]), int(img_info["height"])
        stem = Path(file_name).stem

        for r, row_boxes in enumerate(layer1_boxes(width, height, cols, rows)):
            for c, box in enumerate(row_boxes):
                case_id = f"{split}_{stem}_p{r:02d}{c:02d}"
                save_patch(gray, mask, box, images_out, labels_out, case_id)
                case_ids.append(case_id)

        if include_layer2 and cols > 1 and rows > 1:
            for r, row_boxes in enumerate(layer2_boxes(width, height, cols, rows)):
                for c, box in enumerate(row_boxes):
                    case_id = f"{split}_{stem}_b{r:02d}{c:02d}"
                    save_patch(gray, mask, box, images_out, labels_out, case_id)
                    case_ids.append(case_id)

    return case_ids


def build_dataset(
    dataset_id: int,
    source: str,
    cols: int,
    rows: int,
    include_layer2: bool,
    supported_ids: Optional[Set[int]] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[Path, List[str], List[str], List[str]]:
    source_dir = ARCADE_ROOT / source
    name = dataset_name or f"Dataset{dataset_id:03d}_ARCADE_{cols}x{rows}_1c"
    out_root = NNUNET_RAW / name

    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_ts = out_root / "imagesTs"
    labels_ts = out_root / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    train_cases = process_split("train", source_dir, images_tr, labels_tr, cols, rows, include_layer2, supported_ids)
    val_cases = process_split("val", source_dir, images_tr, labels_tr, cols, rows, include_layer2, supported_ids)
    test_cases = process_split("test", source_dir, images_ts, labels_ts, cols, rows, include_layer2, supported_ids)

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

    return out_root, train_cases, val_cases, test_cases


def parse_grid(value: str) -> Tuple[int, int]:
    cols_str, _, rows_str = value.lower().partition("x")
    return int(cols_str), int(rows_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, default=502)
    parser.add_argument("--source", choices=["syntax", "stenosis"], default="syntax")
    parser.add_argument("--grid", type=parse_grid, default="6x6", help="cols x rows, e.g. 6x6")
    parser.add_argument("--layer2", action="store_true", help="also generate border-straddling layer-2 patches")
    parser.add_argument(
        "--classes",
        help="Comma-separated supported category ids to collapse into the foreground mask "
        "(default: every category except 'stenosis')",
    )
    parser.add_argument("--name", help="Override the generated dataset folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cols, rows = args.grid
    supported_ids = {int(x) for x in args.classes.split(",")} if args.classes else None

    out_root, train_cases, val_cases, test_cases = build_dataset(
        dataset_id=args.dataset_id,
        source=args.source,
        cols=cols,
        rows=rows,
        include_layer2=args.layer2,
        supported_ids=supported_ids,
        dataset_name=args.name,
    )

    print(f"Wrote dataset to {out_root}")
    print(f"train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}")


if __name__ == "__main__":
    main()
