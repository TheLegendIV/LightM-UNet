"""Decompose videos in nnUNet_raw/videos into one nnU-Net dataset per video.

Each video -> nnUNet_raw/Dataset{ID}_video_{N}/imagesTs/frame_{NNNN}_0000.png
  * unique frames only (consecutive near-duplicates dropped, first of each run kept),
    numbered chronologically from 0001. Duplicate threshold is per video (DUP_THRESH):
    video_1 has no duplicates (min real diff 0.41), video_2 is 30 fps with every frame
    doubled (dup diffs <=0.43, real diffs >=0.71).
  * grayscale, bilinear-resized to 512x512

Usage: python -I videos_to_datasets.py [--first-id 512]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RAW = Path(__file__).resolve().parent / "nnUNet_raw"
SIZE = 512
DUP_THRESH = {"videoplayback.mp4": 0.0, "videoplayback (1).mp4": 0.6}


def unique_frames(path, dup_thresh):
    """Yield gray frames whose mean abs diff to the previous *decoded* frame exceeds dup_thresh."""
    cap = cv2.VideoCapture(str(path))
    prev = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev is None or np.abs(gray.astype(np.int16) - prev).mean() > dup_thresh:
            yield gray
        prev = gray.astype(np.int16)
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-id", type=int, default=512)
    args = ap.parse_args()

    videos = sorted((RAW / "videos").glob("*.mp4"), key=lambda p: (len(p.stem), p.stem))
    for i, vid in enumerate(videos):
        ds = RAW / f"Dataset{args.first_id + i}_video_{i + 1}"
        out = ds / "imagesTs"
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for n, g in enumerate(unique_frames(vid, DUP_THRESH[vid.name]), 1):
            up = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / f"frame_{n:04d}_0000.png"), up)
        (ds / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "grayscale"},
            "labels": {"background": 0, "vessel": 1},
            "numTraining": 0,
            "file_ending": ".png",
            "name": ds.name,
            "source_video": vid.name,
        }, indent=2))
        print(f"{vid.name} -> {ds.name}: {n} unique frames")


if __name__ == "__main__":
    main()
