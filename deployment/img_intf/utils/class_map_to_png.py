"""Render a class-id segmentation map (as produced by the FINN accelerator /
run.c's on-device argmax) into a color PNG, using the same 5-class palette
as compression/analysis/notebook/preview_predictions.ipynb
(Dataset509_ARCADE_1x1_4c: 0=Background, 1=LAD, 2=RCA, 3=LCX, 4=LM).

Usage:
    python class_map_to_png.py <input.npy|input.raw> [output.png]

.raw inputs are assumed to be a headerless (h, w) uint8 buffer (h=w=256
unless overridden with --size).
"""
import argparse
import os

import numpy as np
from PIL import Image

CLASS_LABELS = ["Background", "LAD", "RCA", "LCX", "LM"]
CLASS_COLORS_HEX = ["#000000", "#e6194b", "#3cb44b", "#4363d8", "#f58231"]


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


CLASS_COLORS_RGB = np.array([_hex_to_rgb(h) for h in CLASS_COLORS_HEX], dtype=np.uint8)


def load_class_map(path: str, size: int) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, size)
    return np.asarray(arr).squeeze()


def class_map_to_rgb(class_map: np.ndarray) -> np.ndarray:
    n_classes = len(CLASS_COLORS_RGB)
    if class_map.max(initial=0) >= n_classes:
        raise ValueError(f"class_map has values up to {class_map.max()}, only {n_classes} colors defined")
    return CLASS_COLORS_RGB[class_map]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Class-id map (.npy or headerless .raw uint8).")
    parser.add_argument("output", nargs="?", help="Output PNG path (defaults to <input>.png).")
    parser.add_argument("--size", type=int, default=256, help="H/W (square) for .raw inputs.")
    args = parser.parse_args()

    class_map = load_class_map(args.input, args.size)
    rgb = class_map_to_rgb(class_map)

    out_path = args.output or os.path.splitext(args.input)[0] + ".png"
    Image.fromarray(rgb, mode="RGB").save(out_path)

    present = sorted(np.unique(class_map).tolist())
    print(f"Wrote {out_path} (shape={class_map.shape}, classes present={present}: "
          f"{[CLASS_LABELS[c] for c in present]})")


if __name__ == "__main__":
    main()
