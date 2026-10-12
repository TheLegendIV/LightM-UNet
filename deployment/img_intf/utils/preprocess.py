"""Host-side preprocessing for the S12_dense_512 board input.

Reproduces, without torch, what the PyTorch model does before the first conv:
per-image z-score (nnU-Net ZScoreNormalization, whole image) followed by the
model's INT6 input quantizer (scale INPUT_SCALE, zero point 0, signed range
[-32, 31]), then +128 so the level fits the board's UINT8 input.
"""
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SCALE = 0.12401283532381058  # initial.input_quant scale of S12_dense_512_u4_analytical_v1 ft15ep
Q_MIN, Q_MAX = -32, 31
U8_OFFSET = 128


def zscore(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.mean()) / max(float(img.std()), 1e-8)


def u8_from_array(img: np.ndarray) -> np.ndarray:
    q = np.clip(np.round(zscore(img) / np.float32(INPUT_SCALE)), Q_MIN, Q_MAX).astype(np.int32)
    return (q + U8_OFFSET).astype(np.uint8)


def png_to_u8(path) -> np.ndarray:
    return u8_from_array(np.asarray(Image.open(path).convert("L")))


def case_id(path) -> str:
    """test_1_p0000_0000.png -> test_1_p0000 (the label file's stem)."""
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_0000") else stem
