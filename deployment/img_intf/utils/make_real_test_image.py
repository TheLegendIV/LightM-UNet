"""Convert a REAL nnU-Net preprocessed case (already z-score normalized by
nnU-Net's own ZScoreNormalization, see
enet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py)
into a raw 256x256 uint8 test image for the batch marker smoke test.

The preprocessed .npy files under data/nnUNet_preprocessed/.../nnUNetPlans_2d
are (1, 1, H, W) float32 arrays with mean~0, std~1 (verified: e.g. train_1's
min/max are -2.07/+2.75), natively 512x512 (NATIVE_SIZE) -- the ARCADE
dataset's fixed resolution, independent of any particular network's input
size. This is the network's REAL training/inference input distribution --
NOT raw 0-255 pixel intensities. Resized down to SIZE=256 (PIL LANCZOS, after
uint8 quantization) to match the S12_dense_256 build's accelerator input.

We do NOT have the trained checkpoint's actual input QuantIdentity
scale/zero-point available in this (torch/brevitas-free) environment, so
this uses a standard, clearly-approximate float->uint8 mapping instead:
clip to +/-CLIP_STD std devs (covers ~99.7% of a normal distribution at
CLIP_STD=3) then linearly rescale to [0, 255]. This is NOT bit-exact with
the accelerator's real learned input quantizer -- it's a reasonable stand-in
real image for protocol/pipeline smoke testing, not for accuracy validation.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESSED_DIR = (
    REPO_ROOT / "data" / "nnUNet_preprocessed" / "Dataset509_ARCADE_1x1_4c" / "nnUNetPlans_2d"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "input"
CLIP_STD = 3.0
NATIVE_SIZE = 512
SIZE = 256


def zscore_to_uint8(arr: np.ndarray, clip_std: float = CLIP_STD) -> np.ndarray:
    clipped = np.clip(arr, -clip_std, clip_std)
    rescaled = (clipped + clip_std) / (2 * clip_std) * 255.0
    return np.round(rescaled).astype(np.uint8)


def main(case_name: str = "train_1_p0000.npy"):
    src_path = PREPROCESSED_DIR / case_name
    arr = np.load(src_path).squeeze()  # (1,1,H,W) -> (H,W)
    assert arr.shape == (NATIVE_SIZE, NATIVE_SIZE), f"{case_name}: shape {arr.shape}, expected ({NATIVE_SIZE},{NATIVE_SIZE})"

    img_u8 = zscore_to_uint8(arr)
    img_u8 = np.asarray(Image.fromarray(img_u8).resize((SIZE, SIZE), Image.LANCZOS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = os.path.splitext(case_name)[0]
    out_path = OUT_DIR / f"real_zscore_{stem}_{SIZE}x{SIZE}.raw"
    with open(out_path, "wb") as f:
        f.write(img_u8.tobytes())

    print(f"Source: {src_path} (z-normed float32, min={arr.min():.3f} max={arr.max():.3f})")
    print(f"Wrote {img_u8.nbytes} bytes to {out_path} (uint8, min={img_u8.min()} max={img_u8.max()}, resized {NATIVE_SIZE}->{SIZE})")


if __name__ == "__main__":
    main()
