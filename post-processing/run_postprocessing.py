"""Post-processing test bench.

Takes imagesTs (raw frames) and labelsPr_<model_name> (that model's
predictions) from Dataset501_ARCADE, runs postprocess.enhance_prediction on
each pair, and writes the result to labelsPr_<model_name>_Pp/ inside this
folder -- deliberately NOT under data/nnUNet_raw/Dataset501_ARCADE, so
post-processed output never gets confused with a real model's raw prediction.

No ground truth is used here: post-processing only ever sees what a model
would actually have at inference time (the image and its own prediction).
Comparing against labelsTs happens later, in analysis/, by pointing that
tooling at this folder's output instead of a labelsPr_* directory.

Usage:
    python post-processing/run_postprocessing.py                  # all labelsPr_* dirs
    python post-processing/run_postprocessing.py ENetE1 LMUNet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import segmentation_topology as topo

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset501_ARCADE"
IMAGES_TS_DIR = DATASET_DIR / "imagesTs"
OUTPUT_ROOT = Path(__file__).resolve().parent


def discover_net_names() -> list[str]:
    return sorted(
        p.name[len("labelsPr_"):]
        for p in DATASET_DIR.glob("labelsPr_*")
        if p.is_dir()
    )


def load_raw_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def run_for_model(net_name: str) -> None:
    pred_dir = DATASET_DIR / f"labelsPr_{net_name}"
    if not pred_dir.exists():
        print(f"Skipping {net_name}: {pred_dir} does not exist")
        return

    image_paths = {topo.case_id_from_image(p): p for p in topo.image_files(IMAGES_TS_DIR)}
    pred_paths = {p.stem: p for p in topo.image_files(pred_dir)}
    case_ids = sorted(set(image_paths) & set(pred_paths))
    if not case_ids:
        print(f"Skipping {net_name}: no matched imagesTs/{pred_dir.name} pairs")
        return

    out_dir = OUTPUT_ROOT / f"labelsPr_{net_name}_Pp"
    out_dir.mkdir(exist_ok=True)

    for case_id in case_ids:
        image = load_raw_image(image_paths[case_id])
        pred = topo.load_class_id_mask(pred_paths[case_id])
        pred = topo.resize_mask_to(pred, image.shape)
        enhanced = postprocess.enhance_prediction(image, pred)
        Image.fromarray(enhanced.astype(np.uint8)).save(out_dir / f"{case_id}.png")

    print(f"{net_name}: {len(case_ids)} cases -> {out_dir}")


def main() -> None:
    net_names = sys.argv[1:] or discover_net_names()
    if not net_names:
        print(f"No labelsPr_* folders found under {DATASET_DIR}")
        return
    for net_name in net_names:
        run_for_model(net_name)


if __name__ == "__main__":
    main()
