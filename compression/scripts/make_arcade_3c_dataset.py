"""One-off dataset-derivation script: builds Dataset511_ARCADE_1x1_3c from
the existing Dataset509_ARCADE_1x1_4c by merging LM (label 4) into
background (label 0) -- LAD/RCA/LCX (1/2/3) are left untouched. Images are
identical between the two datasets (only the label PNGs differ), so this
copies imagesTr/imagesTs verbatim and only rewrites labelsTr/labelsTs.

Purpose: re-train the ENet-paper-faithful config (channels=20,72,144,72,20,
decoder_type=max_unpool, prelu=1 -- nnUNetTrainerENet.py's own defaults) on
this 3-class variant, to see whether dropping LM changes per-class Dice for
RCA/LCX/LAD (LM is the smallest/rarest class -- ARCADE's own known class-
imbalance outlier -- so it may be soaking up mis-segmented pixels near the
other three vessels' boundaries).

Usage: python compression/scripts/make_arcade_3c_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset509_ARCADE_1x1_4c"
DST = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset511_ARCADE_1x1_3c"

LM_LABEL = 4
BACKGROUND_LABEL = 0


def remap_labels(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for i, f in enumerate(files):
        arr = np.array(Image.open(f))
        arr = np.where(arr == LM_LABEL, BACKGROUND_LABEL, arr).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(dst_dir / f.name)
        if (i + 1) % 200 == 0:
            print(f"  {dst_dir.name}: {i + 1}/{len(files)}")
    print(f"  {dst_dir.name}: {len(files)}/{len(files)} done")


def main() -> None:
    if DST.exists():
        raise SystemExit(f"{DST} already exists -- refusing to overwrite. Delete it first if you want to rebuild.")
    DST.mkdir(parents=True)

    print("Copying imagesTr (unchanged) ...")
    shutil.copytree(SRC / "imagesTr", DST / "imagesTr")
    print("Copying imagesTs (unchanged) ...")
    shutil.copytree(SRC / "imagesTs", DST / "imagesTs")

    print("Remapping labelsTr (LM -> background) ...")
    remap_labels(SRC / "labelsTr", DST / "labelsTr")
    print("Remapping labelsTs (LM -> background) ...")
    remap_labels(SRC / "labelsTs", DST / "labelsTs")

    print("Copying splits_final.json (same case IDs, unaffected by label remap) ...")
    shutil.copy2(SRC / "splits_final.json", DST / "splits_final.json")

    src_json = json.loads((SRC / "dataset.json").read_text())
    dataset_json = {
        "channel_names": src_json["channel_names"],
        "labels": {"background": 0, "LAD": 1, "RCA": 2, "LCX": 3},
        "numTraining": src_json["numTraining"],
        "file_ending": src_json["file_ending"],
        "name": "Dataset511_ARCADE_1x1_3c",
    }
    (DST / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print("Wrote dataset.json:", dataset_json["labels"])
    print("Done:", DST)


if __name__ == "__main__":
    main()
