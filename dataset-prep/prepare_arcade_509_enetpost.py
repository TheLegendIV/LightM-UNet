#!/usr/bin/env python
"""Build Dataset509_ARCADE_ENetPost: a 2-channel, 501-like dataset for
training ENetPost (nnUNetTrainerENetPost) -- ENetOriginal's own
architecture, fed the raw image *plus* a second channel derived from
nnUNetTrainerSmallENet trained on Dataset507_ARCADE_refinement (channel-0
baseline; see analysis/Dataset507_ARCADE_refinement/
preview_507_smallenet_grid_reconstruction.ipynb, which found this
checkpoint's confidence-blend reconstruction beats SmallENet trained
directly on the grid datasets -- 0.587 dice vs 0.583/0.579/0.528 for
502/504/505 respectively).

For every Dataset501_ARCADE image (imagesTr: 1200 train+val, imagesTs: 300
test), this:
  1. Tiles the full image into a 6x6 grid (`layer1_boxes`) + 5x5
     border-straddling overlay (`layer2_boxes`) -- same
     `dataset-prep/patch_grid.py` geometry Dataset502 itself was built
     from, computed directly here rather than depending on Dataset502's
     own patch files (which only exist for its own imagesTs, not every
     Dataset501 image).
  2. Runs each patch through the Dataset507-trained SmallENet checkpoint
     (85x85-native architecture, but resolution-agnostic -- see
     SmallENet.py -- so grid-tile sizes from `split_boundaries` work fine
     even though they aren't exactly 85x85).
  3. Confidence-blend reconstructs the patches back into one full-size
     probability map (`confidence_weight_map`, same RING_KWARGS validated
     in the 507 grid-reconstruction notebook: center_frac=0.2,
     min_confidence=0.5, max_confidence=1.0, falloff="linear",
     metric="chebyshev" -- the view that won on dice in every comparison
     run this session).

Each case gets: {case}_0000.png (Dataset501's original raw image, copied
unchanged), {case}_0001.png (the reconstructed probability map, uint8
0-255), and {case}.png in labelsTr/labelsTs (Dataset501's original 4-class
GT, copied unchanged). channel_names: {"0": "grayscale", "1":
"smallenet507_probability"}. Same splits_final.json as Dataset501 (same
case IDs, same train/val split) -- this is a straight 2-channel
augmentation of Dataset501, not a new patch/split scheme.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# numpy>=2.0-pickled checkpoint compatibility fix (numpy 1.26.4 environment) --
# same as every preview notebook this session.
import numpy.core.multiarray as _ma

_orig_scalar = _ma.scalar


def _patched_scalar(dtype, obj):
    if not isinstance(dtype, np.dtype):
        try:
            dtype = np.dtype(dtype.type) if hasattr(dtype, "type") else np.dtype(dtype)
        except Exception as e:
            raise TypeError(f"could not coerce {dtype!r} to np.dtype") from e
    return _orig_scalar(dtype, obj)


_ma.scalar = _patched_scalar

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lightm-unet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.SmallENet import SmallENet  # noqa: E402
from patch_grid import layer1_boxes, layer2_boxes  # noqa: E402

SOURCE_NAME = "Dataset501_ARCADE"
SOURCE_DIR = REPO_ROOT / "data" / "nnUNet_raw" / SOURCE_NAME
SMALLENET_CKPT = (
    REPO_ROOT / "data" / "nnUNET_results" / "Dataset507_ARCADE_refinement"
    / "nnUNetTrainerSmallENet__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
)

GRID_COLS, GRID_ROWS = 6, 6
RING_KWARGS = dict(center_frac=0.2, min_confidence=0.5, max_confidence=1.0, falloff="linear", metric="chebyshev")

# Must match SmallENet defaults used to train the Dataset507 checkpoint
# (train_small_enet_507.job left these at default).
INITIAL_CHANNELS = 16
STAGE_CHANNELS = 32
LCN_KERNEL_SIZE = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    mean = image.mean()
    std = image.std()
    return (image - mean) / max(std, 1e-8)


def confidence_weight_map(
    shape: tuple[int, int],
    center_frac: float = 0.2,
    min_confidence: float = 0.5,
    max_confidence: float = 1.0,
    falloff: str = "linear",
    metric: str = "chebyshev",
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dy, dx = np.abs(yy - cy), np.abs(xx - cx)

    if metric == "chebyshev":
        dist = np.maximum(dy, dx)
        max_dist = max(cy, cx, 1e-8)
    elif metric == "euclidean":
        dist = np.sqrt(dy ** 2 + dx ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2) + 1e-8
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
    d = np.clip(dist / max_dist, 0.0, 1.0)

    d0 = np.sqrt(np.clip(center_frac, 0.0, 1.0))
    t = np.clip((d - d0) / max(1.0 - d0, 1e-8), 0.0, 1.0)
    if falloff == "linear":
        curve = t
    elif falloff == "gaussian":
        curve = 1.0 - np.exp(-3.0 * t ** 2)
        curve /= (1.0 - np.exp(-3.0))
    else:
        raise ValueError(f"Unknown falloff: {falloff!r}")

    weight = max_confidence - curve * (max_confidence - min_confidence)
    return weight.astype(np.float32)


def load_smallenet(checkpoint_path: Path) -> torch.nn.Module:
    model = SmallENet(
        in_channels=1, out_channels=1,
        initial_channels=INITIAL_CHANNELS, stage_channels=STAGE_CHANNELS, lcn_kernel_size=LCN_KERNEL_SIZE,
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["network_weights"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def reconstruct_probability_map(model: torch.nn.Module, raw: np.ndarray) -> np.ndarray:
    """Grid + overlay confidence-blend reconstruction -- see module docstring."""
    height, width = raw.shape
    boxes1 = layer1_boxes(width, height, GRID_COLS, GRID_ROWS)
    boxes2 = layer2_boxes(width, height, GRID_COLS, GRID_ROWS)
    all_boxes = [box for row in boxes1 for box in row] + [box for row in boxes2 for box in row]

    weighted_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for x0, y0, x1, y1 in all_boxes:
        patch = raw[y0:y1, x0:x1]
        normed = zscore_normalize(patch)
        x = torch.from_numpy(normed).float()[None, None].to(DEVICE)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        w = confidence_weight_map(prob.shape, **RING_KWARGS)
        weighted_sum[y0:y1, x0:x1] += prob * w
        weight_sum[y0:y1, x0:x1] += w

    return weighted_sum / np.maximum(weight_sum, 1e-8)


def prob_to_u8(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob * 255.0, 0, 255).astype(np.uint8)


def process_split(model: torch.nn.Module, images_src: Path, labels_src: Path, images_out: Path, labels_out: Path) -> int:
    img_files = sorted(images_src.glob("*_0000.png"))
    n = 0
    for i, img_path in enumerate(img_files):
        case_id = img_path.stem[: -len("_0000")]
        gt_path = labels_src / f"{case_id}.png"
        if not gt_path.exists():
            print(f"  WARNING: no label for {case_id}, skipping")
            continue

        raw = load_gray(img_path)
        prob = reconstruct_probability_map(model, raw)

        shutil.copyfile(img_path, images_out / f"{case_id}_0000.png")
        Image.fromarray(prob_to_u8(prob), mode="L").save(images_out / f"{case_id}_0001.png")
        shutil.copyfile(gt_path, labels_out / f"{case_id}.png")
        n += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(img_files)} done")
    return n


def build_dataset(dataset_id: int = 509, dataset_name: str | None = None) -> dict:
    name = dataset_name or f"Dataset{dataset_id:03d}_ARCADE_ENetPost"
    out_root = REPO_ROOT / "data" / "nnUNet_raw" / name
    images_tr, labels_tr = out_root / "imagesTr", out_root / "labelsTr"
    images_ts, labels_ts = out_root / "imagesTs", out_root / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    if not SMALLENET_CKPT.exists():
        raise FileNotFoundError(f"{SMALLENET_CKPT} not found -- train nnUNetTrainerSmallENet on Dataset507 first.")

    print(f"Loading SmallENet checkpoint from {SMALLENET_CKPT}")
    model = load_smallenet(SMALLENET_CKPT)

    print("Processing imagesTr/labelsTr (train+val)...")
    n_tr = process_split(model, SOURCE_DIR / "imagesTr", SOURCE_DIR / "labelsTr", images_tr, labels_tr)
    print(f"  {n_tr} train+val cases written")

    print("Processing imagesTs/labelsTs (test)...")
    n_ts = process_split(model, SOURCE_DIR / "imagesTs", SOURCE_DIR / "labelsTs", images_ts, labels_ts)
    print(f"  {n_ts} test cases written")

    shutil.copyfile(SOURCE_DIR / "splits_final.json", out_root / "splits_final.json")

    dataset_json = {
        "channel_names": {"0": "grayscale", "1": "smallenet507_probability"},
        "labels": {"background": 0, "LAD": 1, "RCA": 2, "LCX": 3},
        "numTraining": n_tr,
        "file_ending": ".png",
        "name": name,
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    return {"out_root": out_root, "n_train": n_tr, "n_test": n_ts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=509)
    parser.add_argument("--name", help="Override the generated dataset folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(dataset_id=args.dataset_id, dataset_name=args.name)
    print(f"Wrote dataset to {result['out_root']}")
    print(f"train+val={result['n_train']} test={result['n_test']}")


if __name__ == "__main__":
    main()
