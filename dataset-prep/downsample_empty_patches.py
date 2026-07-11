#!/usr/bin/env python
"""Build a copy of an ARCADE patch dataset with empty (all-background) training
patches randomly downsampled to a target vessel:empty ratio.

Only the *training* pool (the "train" half of splits_final.json) is
downsampled -- the "val" half is left at its natural (imbalanced) distribution
so per-epoch validation dice stays comparable across experiments, and the test
split (imagesTs/labelsTs) is not copied at all since training/preprocessing
don't need it (point evaluation notebooks at the source dataset's Ts folder --
it is untouched and identical).

Usage:
    python dataset-prep/downsample_empty_patches.py \
        --source-dataset-id 502 --dataset-id 503 --ratio 2.0 --seed 0

Then, same as any freshly built raw dataset:
    nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
NNUNET_RAW = REPO_ROOT / "data" / "nnUNet_raw"


def vessel_pixel_counts(labels_tr_dir: Path) -> dict[str, int]:
    """case_id -> count of foreground (==1) pixels in its label mask."""
    counts = {}
    for path in sorted(labels_tr_dir.glob("*.png")):
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        counts[path.stem] = int((arr == 1).sum())
    return counts


def load_vessel_pixel_counts_csv(csv_path: Path) -> dict[str, int]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    case_ids = df["file"].str.replace(".png", "", regex=False)
    return dict(zip(case_ids, df["vessel_px"]))


def build(
    source_dataset_id: int,
    dataset_id: int,
    ratio: float,
    seed: int,
    name: str | None,
    counts_csv: Path | None,
) -> Path:
    source_dir = next(NNUNET_RAW.glob(f"Dataset{source_dataset_id:03d}_*"))
    dataset_json = json.loads((source_dir / "dataset.json").read_text())
    splits = json.loads((source_dir / "splits_final.json").read_text())
    assert len(splits) == 1, "Expected a single-fold splits_final.json."
    train_ids, val_ids = list(splits[0]["train"]), list(splits[0]["val"])

    counts = (
        load_vessel_pixel_counts_csv(counts_csv) if counts_csv is not None
        else vessel_pixel_counts(source_dir / "labelsTr")
    )
    missing = [c for c in train_ids + val_ids if c not in counts]
    if missing:
        raise ValueError(f"{len(missing)} case ids missing from vessel-pixel counts, e.g. {missing[:5]}")

    train_vessel = [c for c in train_ids if counts[c] > 0]
    train_empty = [c for c in train_ids if counts[c] == 0]

    n_empty_target = round(len(train_vessel) / ratio)
    n_empty_target = min(n_empty_target, len(train_empty))
    rng = random.Random(seed)
    train_empty_kept = rng.sample(train_empty, n_empty_target)

    new_train_ids = sorted(train_vessel + train_empty_kept)
    kept_ids = set(new_train_ids) | set(val_ids)

    name = name or f"Dataset{dataset_id:03d}_ARCADE_6x6_1c_ve{ratio:g}to1"
    out_root = NNUNET_RAW / name
    images_tr_out = out_root / "imagesTr"
    labels_tr_out = out_root / "labelsTr"
    images_tr_out.mkdir(parents=True, exist_ok=True)
    labels_tr_out.mkdir(parents=True, exist_ok=True)

    for case_id in sorted(kept_ids):
        shutil.copy2(source_dir / "imagesTr" / f"{case_id}_0000.png", images_tr_out / f"{case_id}_0000.png")
        shutil.copy2(source_dir / "labelsTr" / f"{case_id}.png", labels_tr_out / f"{case_id}.png")

    new_dataset_json = {**dataset_json, "numTraining": len(kept_ids), "name": name}
    (out_root / "dataset.json").write_text(json.dumps(new_dataset_json, indent=2))

    new_splits = [{"train": new_train_ids, "val": sorted(val_ids)}]
    (out_root / "splits_final.json").write_text(json.dumps(new_splits, indent=2))

    print(f"Source           : {source_dir.name}")
    print(f"Output           : {out_root}")
    print(f"train vessel     : {len(train_vessel)}")
    print(f"train empty kept : {len(train_empty_kept)} / {len(train_empty)} (target ratio {ratio:g}:1)")
    print(f"train total      : {len(new_train_ids)}")
    print(f"val (unchanged)  : {len(val_ids)}")
    print(f"numTraining      : {len(kept_ids)}")
    print("Note: imagesTs/labelsTs were NOT copied -- point evaluation at the source "
          f"dataset's Ts folder ({source_dir / 'imagesTs'}); test patches are identical.")
    return out_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-id", type=int, default=502)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--ratio", type=float, default=2.0, help="Target vessel:empty ratio in the training pool.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="Override the generated dataset folder name")
    parser.add_argument(
        "--counts-csv", type=Path, default=None,
        help="Precomputed per-case vessel pixel counts CSV (file,vessel_px columns), "
        "e.g. analysis/502_ARCADE_6x6_1c/results/Dataset502_ARCADE_6x6_1c_gt_vessel_pixel_counts.csv. "
        "If omitted, recomputes by scanning labelsTr.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(
        source_dataset_id=args.source_dataset_id,
        dataset_id=args.dataset_id,
        ratio=args.ratio,
        seed=args.seed,
        name=args.name,
        counts_csv=args.counts_csv,
    )


if __name__ == "__main__":
    main()
