"""Derive Dataset511_ARCADE_1x1_3c from Dataset509_ARCADE_1x1_4c by merging LM (label 4) into background."""
import json
import os
import shutil

import numpy as np
from PIL import Image

from nnunetv2.paths import nnUNet_raw

SRC = os.path.join(nnUNet_raw, "Dataset509_ARCADE_1x1_4c")
DST = os.path.join(nnUNet_raw, "Dataset511_ARCADE_1x1_3c")
DROPPED_LABEL = 4


def main():
    for sub in ("imagesTr", "imagesTs"):
        shutil.copytree(os.path.join(SRC, sub), os.path.join(DST, sub), dirs_exist_ok=True)

    for sub in ("labelsTr", "labelsTs"):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)
        for name in sorted(os.listdir(os.path.join(SRC, sub))):
            img = Image.open(os.path.join(SRC, sub, name))
            seg = np.array(img)
            seg[seg == DROPPED_LABEL] = 0
            if seg.max() > 3:
                raise ValueError(f"{sub}/{name}: unexpected label {seg.max()}")
            Image.fromarray(seg, mode=img.mode).save(os.path.join(DST, sub, name))

    shutil.copy(os.path.join(SRC, "splits_final.json"), os.path.join(DST, "splits_final.json"))

    with open(os.path.join(SRC, "dataset.json")) as f:
        dataset_json = json.load(f)
    dataset_json["labels"] = {"background": 0, "LAD": 1, "RCA": 2, "LCX": 3}
    dataset_json["name"] = "Dataset511_ARCADE_1x1_3c"
    with open(os.path.join(DST, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)


if __name__ == "__main__":
    main()
