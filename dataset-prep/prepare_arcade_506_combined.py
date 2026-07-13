#!/usr/bin/env python
"""Build Dataset506_stn_stx_oversampled = Dataset505_ARCADE_oversampled (the
"syntax" source, prefixed "stx_") plus a matching vessel-coverage build of
the "stenosis" source (prefixed "stn_").

The two ARCADE sources annotate differently: "syntax" labels the full
per-segment vessel tree (prepare_arcade_vessel_coverage.py's default
category exclusion -- everything except the "stenosis" category -- collapses
those segments into one binary vessel mask). "stenosis" annotates *only* the
narrowed lesion region itself (checked directly against this repo's copy:
1,625 annotations across 1000 train images, 100% category "stenosis", median
1/image, some images with zero) -- there is no separate vessel-segment label
to fall back on there. A stenotic segment is still vessel, just a different
and more localized annotation, so here the "stenosis" category *is* the
vessel mask (the opposite of syntax's exclusion default) -- pass
--classes with that source's "stenosis" category id explicitly.

Same skeletonize + farthest-point-sample vessel-coverage patch extraction as
Dataset505 (prepare_arcade_vessel_coverage.py) is applied to both sources;
this script's only job is running that twice with a distinguishing
case_prefix (avoids id collisions -- both sources independently number their
images from 1, so e.g. "234.png" exists in both and is unrelated) and
merging the results into one dataset.json/splits_final.json. Everything
about a patch's provenance is recoverable from its case id prefix
("stx_"/"stn_") after the merge.

Run prepare_arcade_vessel_coverage.py to (re)build Dataset505 first if it
doesn't already exist at data/nnUNet_raw/Dataset505_ARCADE_oversampled --
this script copies its patches rather than re-deriving them from source.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco2png import load_coco
from prepare_arcade import default_supported_classes
from prepare_arcade_vessel_coverage import ARCADE_ROOT, NNUNET_RAW, PATCH_SIZE, PATCHES_PER_IMAGE, process_split

STX_PREFIX = "stx_"
STN_PREFIX = "stn_"
STENOSIS_CATEGORY_NAME = "stenosis"


def copy_dataset505_with_prefix(images_tr: Path, labels_tr: Path, prefix: str) -> Tuple[List[str], List[str]]:
    src_root = NNUNET_RAW / "Dataset505_ARCADE_oversampled"
    if not src_root.exists():
        raise FileNotFoundError(
            f"{src_root} not found -- build Dataset505 first "
            "(python prepare_arcade_vessel_coverage.py --dataset-id 505)."
        )
    splits = json.loads((src_root / "splits_final.json").read_text())
    train_cases_505 = splits[0]["train"]
    val_cases_505 = splits[0]["val"]

    train_cases, val_cases = [], []
    for cases_505, out_list in ((train_cases_505, train_cases), (val_cases_505, val_cases)):
        for case_id in cases_505:
            new_case_id = f"{prefix}{case_id}"
            shutil.copyfile(src_root / "imagesTr" / f"{case_id}_0000.png", images_tr / f"{new_case_id}_0000.png")
            shutil.copyfile(src_root / "labelsTr" / f"{case_id}.png", labels_tr / f"{new_case_id}.png")
            out_list.append(new_case_id)
    print(f"  copied Dataset505 (stx): train={len(train_cases)} val={len(val_cases)}")
    return train_cases, val_cases


def stenosis_category_id(source_dir: Path) -> int:
    coco = load_coco(source_dir / "train" / "annotations" / "train.json")
    for c in coco.get("categories", []):
        if c.get("name") == STENOSIS_CATEGORY_NAME:
            return int(c["id"])
    raise ValueError(f"no '{STENOSIS_CATEGORY_NAME}' category found in {source_dir}")


def build_dataset506(dataset_id: int = 506, patch_size: int = PATCH_SIZE, patches_per_image: int = PATCHES_PER_IMAGE):
    name = f"Dataset{dataset_id:03d}_stn_stx_oversampled"
    out_root = NNUNET_RAW / name
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    for d in (images_tr, labels_tr):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Building {name}...")
    stx_train, stx_val = copy_dataset505_with_prefix(images_tr, labels_tr, STX_PREFIX)

    stenosis_source_dir = ARCADE_ROOT / "stenosis"
    stenosis_ids = {stenosis_category_id(stenosis_source_dir)}
    print(f"  stenosis source: using category id(s) {stenosis_ids} ('{STENOSIS_CATEGORY_NAME}') as the vessel mask")
    stn_train = process_split(
        "train", stenosis_source_dir, images_tr, labels_tr, patch_size, patches_per_image, stenosis_ids, STN_PREFIX
    )
    stn_val = process_split(
        "val", stenosis_source_dir, images_tr, labels_tr, patch_size, patches_per_image, stenosis_ids, STN_PREFIX
    )

    train_cases = stx_train + stn_train
    val_cases = stx_val + stn_val

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

    return out_root, train_cases, val_cases, stx_train, stx_val, stn_train, stn_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=506)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patches-per-image", type=int, default=PATCHES_PER_IMAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root, train_cases, val_cases, stx_train, stx_val, stn_train, stn_val = build_dataset506(
        dataset_id=args.dataset_id, patch_size=args.patch_size, patches_per_image=args.patches_per_image
    )
    print(f"Wrote dataset to {out_root}")
    print(f"train={len(train_cases)} (stx={len(stx_train)} + stn={len(stn_train)})")
    print(f"val={len(val_cases)} (stx={len(stx_val)} + stn={len(stn_val)})")
    print(
        "Note: no imagesTs/labelsTs generated -- point evaluation at "
        "Dataset502_ARCADE_6x6_1c's imagesTs/labelsTs (standard grid tiling, not vessel-biased)."
    )


if __name__ == "__main__":
    main()
