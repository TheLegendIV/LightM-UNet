"""ARCADE test-split sample: the 512x512 grayscale angiogram with the four vessel-class masks overlaid, no axes or text (the figure shows the segmentation problem).
Colours and blending = the img_intf streamer's own overlay (deployment/img_intf/utils/raw_to_png.py, used by stream_images.py --overlay), imported so they cannot drift: LAD red, RCA green, LCX blue, LM yellow, alpha 0.5 (Dataset509_ARCADE_1x1_4c label ids 1..4).
No test image (nor any ARCADE image in the 300-case test split) contains all four classes: RCA never co-occurs with LAD / LCX / LM (right- vs left-coronary views). So two images cover the four classes:
  left  = the test case with LAD + LCX + LM present whose smallest of the three has the most pixels (all three clearly visible)
  right = the test case with the most RCA pixels
--case overrides (one image).

Run in lightmunet_dev:   docker exec lightmunet_dev python3 /workspace/LightM-UNet/report/sw/dataset_overlay/plot_arcade_overlay.py [--case test_100_p0000]
"""
import argparse
from pathlib import Path

import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "deployment"))
from img_intf.utils.raw_to_png import overlay  # noqa: E402

RAW = REPO_ROOT / "data" / "nnUNet_raw" / "Dataset509_ARCADE_1x1_4c"
HERE = Path(__file__).resolve().parent
CLASSES = ["LAD", "RCA", "LCX", "LM"]


def class_pixels(label):
    return [int((label == i).sum()) for i in range(1, 5)]


def pick_cases():
    """(left-system case, RCA case); prints the top 3 of each ranking."""
    left, right, n_all4 = [], [], 0
    for p in sorted((RAW / "labelsTs").glob("*.png")):
        c = class_pixels(np.array(Image.open(p)))
        n_all4 += all(c)
        if c[0] and c[2] and c[3]:                      # LAD, LCX, LM
            left.append((min(c[0], c[2], c[3]), p.stem, c))
        if c[1]:
            right.append((c[1], p.stem, c))
    left.sort(reverse=True)
    right.sort(reverse=True)
    print(f"test cases with all four classes: {n_all4}")
    for title, rows in (("LAD+LCX+LM, by smallest class", left), ("RCA, by RCA pixels", right)):
        print(f"{len(rows)} cases, top 3 {title} ({'/'.join(CLASSES)} px):")
        for _, name, counts in rows[:3]:
            print(f"  {name}: {counts}")
    return left[0][1], right[0][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None, help="test case name, e.g. test_100_p0000 (default: see above)")
    ap.add_argument("--alpha", type=float, default=0.5, help="mask opacity (the streamer's default)")
    ap.add_argument("--out", type=Path, default=None, help="output path (only with --case)")
    a = ap.parse_args()

    for case in [a.case] if a.case else pick_cases():
        img = np.array(Image.open(RAW / "imagesTs" / f"{case}_0000.png").convert("L"))
        label = np.array(Image.open(RAW / "labelsTs" / f"{case}.png"))
        assert img.shape == label.shape == (512, 512), (img.shape, label.shape)

        path = a.out or HERE / f"arcade_test_overlay_{case}.png"
        Image.fromarray(overlay(img, label, a.alpha)).save(path)
        print(f"{case}: pixels per class {dict(zip(CLASSES, class_pixels(label)))}")
        print(f"wrote {path} ({Image.open(path).size[0]}x{Image.open(path).size[1]})")


if __name__ == "__main__":
    main()
