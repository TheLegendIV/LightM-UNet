"""Build a raw 256x256 uint8 test image directly from a REAL ARCADE test-split
PNG (data/nnUNet_raw/Dataset509_ARCADE_1x1_4c/imagesTs/*.png), applying the
SAME per-image z-score normalization nnU-Net's own preprocessor uses for this
dataset (confirmed in nnUNetPlans.json: normalization_schemes=["ZScoreNormalization"],
use_mask_for_norm=[false] -> plain whole-image (image - image.mean()) / image.std(),
see enet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py's
ZScoreNormalization.run()).

This differs from make_real_test_image.py (which starts from an ALREADY
z-normed .npy cached by nnU-Net's preprocessing pipeline) -- this script starts
from the original 0-255 PNG and reproduces the z-score step ourselves, so you
can point it at any imagesTs case instead of only ones already preprocessed.
The source PNGs are natively NATIVE_SIZE (512x512); resized down to SIZE=256
(PIL LANCZOS, after uint8 quantization) to match the S12_dense_256 build.

Same caveat as make_real_test_image.py: converting the resulting float
z-scored array back to uint8 (clip +/-CLIP_STD std devs, rescale to [0,255])
is a reasonable stand-in for protocol/pipeline smoke testing, NOT bit-exact
with the accelerator's real learned input quantizer.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from make_real_test_image import zscore_to_uint8, CLIP_STD, NATIVE_SIZE, SIZE

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_TS_DIR = (
    REPO_ROOT / "data" / "nnUNet_raw" / "Dataset509_ARCADE_1x1_4c" / "imagesTs"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "input"


def main(png_name: str = "test_1_p0000_0000.png"):
    src_path = IMAGES_TS_DIR / png_name
    img = np.array(Image.open(src_path).convert("L"), dtype=np.float32)
    assert img.shape == (NATIVE_SIZE, NATIVE_SIZE), f"{png_name}: shape {img.shape}, expected ({NATIVE_SIZE},{NATIVE_SIZE})"

    # nnU-Net ZScoreNormalization.run(), use_mask_for_norm=False branch.
    mean, std = img.mean(), img.std()
    zscored = (img - mean) / max(std, 1e-8)

    img_u8 = zscore_to_uint8(zscored)
    img_u8 = np.asarray(Image.fromarray(img_u8).resize((SIZE, SIZE), Image.LANCZOS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(png_name).stem
    out_path = OUT_DIR / f"real_zscore_from_png_{stem}_{SIZE}x{SIZE}.raw"
    with open(out_path, "wb") as f:
        f.write(img_u8.tobytes())

    print(f"Source: {src_path} (raw uint8, min={img.min():.0f} max={img.max():.0f})")
    print(f"Z-scored: mean={mean:.3f} std={std:.3f} -> min={zscored.min():.3f} max={zscored.max():.3f} (clip=+/-{CLIP_STD})")
    print(f"Wrote {img_u8.nbytes} bytes to {out_path} (uint8, min={img_u8.min()} max={img_u8.max()}, resized {NATIVE_SIZE}->{SIZE})")


if __name__ == "__main__":
    main()
