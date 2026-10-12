"""Host-side preprocessing for the S12_dense_512 board input.

Reproduces, without torch, what the PyTorch model does before the first conv:
per-image z-score (nnU-Net ZScoreNormalization, whole image) followed by the
model's signed input quantizer (zero point 0; scale and bit width come from a
named profile in input_profiles.json, one per build), then +128 so the level
fits the board's UINT8 input.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROFILES_JSON = Path(__file__).with_name("input_profiles.json")
DEFAULT_PROFILE = "s12_512_nn"
U8_OFFSET = 128


def load_profile(name: str = DEFAULT_PROFILE) -> tuple[float, int, int]:
    """-> (scale, q_min, q_max) of the named build's input quantizer."""
    profiles = json.loads(PROFILES_JSON.read_text())
    if name not in profiles:
        raise KeyError(f"input profile {name!r} not in {PROFILES_JSON.name}: {sorted(profiles)}")
    bits = profiles[name]["bits"]
    return float(profiles[name]["scale"]), -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def zscore(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.mean()) / max(float(img.std()), 1e-8)


def u8_from_array(img: np.ndarray, profile: str = DEFAULT_PROFILE) -> np.ndarray:
    scale, q_min, q_max = load_profile(profile)
    q = np.clip(np.round(zscore(img) / np.float32(scale)), q_min, q_max).astype(np.int32)
    return (q + U8_OFFSET).astype(np.uint8)


def png_to_u8(path, profile: str = DEFAULT_PROFILE) -> np.ndarray:
    return u8_from_array(np.asarray(Image.open(path).convert("L")), profile)


def case_id(path) -> str:
    """test_1_p0000_0000.png -> test_1_p0000 (the label file's stem)."""
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_0000") else stem
