"""Per-image z-score (nnU-Net ZScoreNormalization, no mask) of deployment/256_test_images.

Writes the full-frame float32 z-score (hardware path) to deployment/256_test_images_zscore/ and compares the
nnU-Net-style result (crop to non-zero bbox, then z-score) against the cases in
data/nnUNet_preprocessed/Dataset510_ARCADE_256_4c.
Run: .venv\\Scripts\\python.exe hardware/temp/_zscore_256_test_images.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization  # noqa: E402

IN_DIR = REPO_ROOT / "deployment" / "256_test_images"
OUT_DIR = REPO_ROOT / "deployment" / "256_test_images_zscore"
PRE_DIR = REPO_ROOT / "data" / "nnUNet_preprocessed" / "Dataset510_ARCADE_256_4c" / "nnUNetPlans_2d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

norm = ZScoreNormalization(use_mask_for_norm=False, intensityproperties={})
pre_files = sorted(p for p in PRE_DIR.glob("*.npy") if not p.stem.endswith("_seg"))
pre = {p.stem: np.load(p, mmap_mode="r")[0, 0] for p in pre_files}


def zscore(a):
    return norm.run(a.astype(np.float32)[None], None)[0].astype(np.float32)


for p in sorted(glob.glob(str(IN_DIR / "*.png"))):
    stem = Path(p).stem
    a = np.asarray(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0]
    z = zscore(a)
    np.save(OUT_DIR / f"{stem}_zscore.npy", z)

    nz = np.argwhere(a != 0)
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
    zc = zscore(a[r0:r1, c0:c1])
    best = min((k for k, v in pre.items() if v.shape == zc.shape), key=lambda k: float(np.abs(pre[k] - zc).max()), default=None)
    d = float(np.abs(pre[best] - zc).max()) if best else float("nan")
    match = f"{best} (max|diff| {d:.2e})" if best and d < 1e-3 else f"no preprocessed match (closest {best}, {d:.2e})"
    print(f"{stem}: shape {a.shape} nonzero-crop {zc.shape} | mean {z.mean():+.1e} std {z.std():.4f} range [{z.min():.2f}, {z.max():.2f}] | {match}")
