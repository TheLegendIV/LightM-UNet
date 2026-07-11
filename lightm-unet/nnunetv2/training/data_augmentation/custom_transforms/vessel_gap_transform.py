from __future__ import annotations

import os

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

"""
Prototyped/visualized in dataset-prep/preview_augmentations.ipynb before being
wired in here -- see that notebook for the reasoning and example images.

Reduces image contrast (does NOT blur, does NOT fully erase) in a small block
centered on a random point along a vessel's skeleton, while leaving the
segmentation label untouched. The cut point is chosen so both sides have a
guaranteed minimum run of unambiguous vessel (not background, not a
bifurcation) -- the goal is to force the network to predict connectivity
through a locally low-evidence patch of the image instead of only tracing
visible contrast.
"""


def _skeleton_degree(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = ndi.convolve(skel.astype(int), kernel, mode="constant") - skel.astype(int)
    return neighbor_count * skel


def _order_path(component_mask: np.ndarray) -> list[tuple[int, int]]:
    """Walk a simple (unbranched) skeleton component end to end."""
    coords = set(map(tuple, np.argwhere(component_mask)))
    deg = {
        p: sum((p[0] + dy, p[1] + dx) in coords for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
        for p in coords
    }
    endpoints = [p for p, d in deg.items() if d == 1]
    start = endpoints[0] if endpoints else next(iter(coords))
    path, visited, cur = [start], {start}, start
    while len(path) < len(coords):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (cur[0] + dy, cur[1] + dx)
                if nb in coords and nb not in visited:
                    nxt = nb
                    break
            if nxt is not None:
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


def _extract_segments(skel: np.ndarray, min_length: int) -> list[list[tuple[int, int]]]:
    """Bifurcation-free vessel runs, each ordered as a walk between its two ends."""
    deg = _skeleton_degree(skel)
    trimmed = skel & (deg < 3)
    labeled, n = ndi.label(trimmed, structure=np.ones((3, 3)))
    segments = []
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() >= min_length:
            segments.append(_order_path(comp))
    return segments


class VesselGapTransform(AbstractTransform):
    """
    Toggle/tune via env vars (all read once, at construction time, via
    VesselGapTransform.from_env() -- see nnUNetTrainerSmallENet):

      SMALLENET_GAP_AUG=1                enable (disabled by default)
      SMALLENET_GAP_AUG_P=0.3            probability of applying to each sample in the batch
      SMALLENET_GAP_AUG_DROP=0.55        contrast drop fraction in the block, 0-1
      SMALLENET_GAP_AUG_MARGIN=8         min skeleton pixels kept on each side of the cut
      SMALLENET_GAP_AUG_MIN_SEGMENT=40   shortest cuttable skeleton segment, in pixels
    """

    def __init__(
        self,
        p_per_sample: float = 0.3,
        drop_frac: float = 0.55,
        margin: int = 8,
        min_segment_length: int = 40,
        data_key: str = "data",
        seg_key: str = "seg",
    ):
        self.p_per_sample = p_per_sample
        self.drop_frac = drop_frac
        self.margin = margin
        self.min_segment_length = min_segment_length
        self.data_key = data_key
        self.seg_key = seg_key

    @classmethod
    def from_env(cls) -> "VesselGapTransform | None":
        if os.environ.get("SMALLENET_GAP_AUG", "0") != "1":
            return None
        return cls(
            p_per_sample=float(os.environ.get("SMALLENET_GAP_AUG_P", "0.3")),
            drop_frac=float(os.environ.get("SMALLENET_GAP_AUG_DROP", "0.55")),
            margin=int(os.environ.get("SMALLENET_GAP_AUG_MARGIN", "8")),
            min_segment_length=int(os.environ.get("SMALLENET_GAP_AUG_MIN_SEGMENT", "40")),
        )

    def _pick_cut_box(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if mask.sum() < self.min_segment_length:
            return None
        skel = skeletonize(mask)
        segments = _extract_segments(skel, self.min_segment_length)
        if not segments:
            return None
        segment = segments[np.random.randint(len(segments))]
        margin = self.margin
        if len(segment) < 2 * margin + 2:
            center = segment[len(segment) // 2]
        else:
            center = segment[np.random.randint(margin, len(segment) - margin)]

        radius = ndi.distance_transform_edt(mask)[center]
        half_width = max(3, int(round(radius)) + 2)
        y, x = center
        h, w = mask.shape
        y0, y1 = max(0, y - half_width), min(h, y + half_width + 1)
        x0, x1 = max(0, x - half_width), min(w, x + half_width + 1)
        return y0, y1, x0, x1

    def _apply_drop(self, image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        y0, y1, x0, x1 = box
        bg = image[~mask]
        local_mean = bg.mean() if bg.size else image.mean()
        out = image.copy()
        block = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = local_mean + (1 - self.drop_frac) * (block - local_mean)
        return out

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        seg = data_dict[self.seg_key]
        for b in range(data.shape[0]):
            if np.random.uniform() >= self.p_per_sample:
                continue
            mask = seg[b, 0] == 1
            box = self._pick_cut_box(mask)
            if box is None:
                continue
            for c in range(data.shape[1]):
                data[b, c] = self._apply_drop(data[b, c], mask, box)
        return data_dict
