"""Diagnostic: hooks every enumerate_blocks() bottleneck in a calibrated,
NOT-yet-QAT'd Quant* model and checks each block's output activation
variance across a handful of real val images. A block whose output is
(near-)constant regardless of input -- zero or near-zero variance -- has
had its signal destroyed by quantization (e.g. an activation quantizer
clipped so tightly everything maps to one bin), which would explain a
literal, unmoving 0.0 pseudo-dice for many epochs straight: no gradient
signal can flow back through a dead block to fix it via ordinary QAT.

Run for both the working control (S19-twopass-validated) and the stuck
config (S19-joint) on the SAME images, so a block that's fine in one but
dead in the other points directly at the bit choice responsible.

Usage (inside the dev container):
    python compression/hawq/diagnose_dead_blocks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
from nnunetv2.nets.QuantENetS19Block import QuantENetS19Block  # noqa: E402

from brevitas.graph.calibrate import calibration_mode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_utils import TOP_LEVEL_ATTRS  # noqa: E402
from torch import nn  # noqa: E402


def enumerate_blocks_tolerant(model) -> dict:
    """Same as block_utils.enumerate_blocks, but skips a TOP_LEVEL_ATTRS
    name entirely if the model has no such attribute at all (proj2_to_3 is
    nn.Identity() when stage2/stage3 channels match, and Quant* subclasses
    sometimes don't register it as a submodule at all in that case, unlike
    plain ENet.py which always has the attribute -- block_utils.enumerate_blocks
    assumes the latter)."""
    blocks = {}
    for attr in TOP_LEVEL_ATTRS:
        if not hasattr(model, attr):
            continue
        module = getattr(model, attr)
        if isinstance(module, nn.Identity):
            continue
        children_source = module.ops if hasattr(module, "ops") and isinstance(module.ops, nn.ModuleList) else module
        children = list(children_source.named_children()) if isinstance(children_source, (nn.Sequential, nn.ModuleList)) else []
        if children:
            for name, child in children:
                blocks[f"{attr}.{name}"] = child
        else:
            blocks[attr] = module
    return blocks

NNUNET_PREPROCESSED = REPO_ROOT / "data" / "nnUNet_preprocessed"
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CALIB_IMAGES = 64
N_PROBE_IMAGES = 8

FP32_CKPT = (
    REPO_ROOT / "data/nnUNet_results" / DATASET_NAME
    / "nnUNetTrainerENet_19_reginterleaved_separable_nonneg_block_double_mid__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_final.pth"
)
SLOPE_MAP_FILE = REPO_ROOT / "compression/post-quantization/slope_maps/19_reginterleaved_separable_nonneg_block_double_mid.json"

RUNS = {
    "twopass_validated": REPO_ROOT / "compression/hawq/block_bits_s19_validated.json",
    "joint": REPO_ROOT / "compression/hawq/block_bits_s19_joint.json",
}


def calibrate(model: torch.nn.Module, n_images: int, seed: int = 0) -> int:
    import random
    d = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans_2d"
    train_files = sorted(p for p in d.glob("train_*.npy") if not p.name.endswith("_seg.npy"))
    rng = random.Random(seed)
    sampled = rng.sample(train_files, k=min(n_images, len(train_files)))
    model.to(DEVICE)
    model.train()
    n_used = 0
    with torch.no_grad(), calibration_mode(model):
        for p in sampled:
            img = torch.from_numpy(np.load(p)).float().to(DEVICE)
            try:
                model(img)
                n_used += 1
            except RuntimeError:
                continue
    model.eval()
    return n_used


def probe_block_variance(model: torch.nn.Module, image_paths: list[Path]) -> dict[str, float]:
    blocks = enumerate_blocks_tolerant(model)
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in blocks}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            out_t = out.value if hasattr(out, "value") else out
            captured[name].append(out_t.detach().float().cpu())
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in blocks.items()]
    with torch.no_grad():
        for p in image_paths:
            img = torch.from_numpy(np.load(p)).float().to(DEVICE)
            try:
                model(img)
            except RuntimeError:
                continue
    for h in handles:
        h.remove()

    # Real preprocessed images vary slightly in H/W (odd-sized inputs), so
    # outputs across images can't be stacked into one tensor. Instead: for
    # each image, compute the per-channel SPATIAL variance (is this feature
    # map constant across pixels, within one image?), average over channels,
    # then average that per-image scalar across images -- a dead/saturated
    # channel (same value at every pixel, every image) drives this to ~0
    # regardless of the exact H/W each image happens to have.
    variances = {}
    for name, outs in captured.items():
        if not outs:
            variances[name] = float("nan")
            continue
        per_image = [out.var(dim=(0, 2, 3)).mean().item() for out in outs]
        variances[name] = float(np.mean(per_image))
    return variances


def main() -> None:
    d = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans_2d"
    val_files = sorted(p for p in d.glob("val_*.npy") if not p.name.endswith("_seg.npy"))
    if not val_files:
        val_files = sorted(p for p in d.glob("train_*.npy") if not p.name.endswith("_seg.npy"))
    probe_images = val_files[:N_PROBE_IMAGES]
    print(f"Probing with {len(probe_images)} real images.\n")

    with open(SLOPE_MAP_FILE) as f:
        slope_map = json.load(f)

    all_variances = {}
    for label, bits_file in RUNS.items():
        with open(bits_file) as f:
            bits = json.load(f)
        model = QuantENetS19Block.from_pretrained(
            FP32_CKPT, bits["stage_weight_bits"], bits["stage_act_bits"], slope_map,
        )
        calibrate(model, N_CALIB_IMAGES)
        variances = probe_block_variance(model, probe_images)
        all_variances[label] = variances
        print(f"=== {label} ===")
        for name, v in variances.items():
            print(f"  {name:20s} var={v:.6e}")
        print()

    print("=== Blocks with near-zero variance under 'joint' but healthy under 'twopass_validated' ===")
    for name in all_variances["joint"]:
        v_joint = all_variances["joint"][name]
        v_twopass = all_variances["twopass_validated"].get(name, float("nan"))
        if v_joint < 1e-8 and v_twopass > 1e-6:
            print(f"  {name}: joint var={v_joint:.3e}  twopass var={v_twopass:.3e}  <-- SUSPECT DEAD BLOCK")


if __name__ == "__main__":
    main()
