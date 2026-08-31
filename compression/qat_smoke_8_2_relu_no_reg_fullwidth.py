"""Quick local QAT smoke test: does nnUNetTrainerENet_8_2_relu_no_reg_
fullwidth train faster at a higher LR (5e-3, 5x the base 0.001 default)
under its own real HAWQ mixed per-stage bit assignment (compression/hawq/
stage_bits_8_2_relu_no_reg_fullwidth_minres.json)?

Same mechanism as compression/qat_smoke_8_2_relu_no_reg_d2_projected.py
(see that file's own module docstring for the full rationale -- not
repeated here): a standalone real-gradient training loop (real preprocessed
ARCADE data + nnU-Net's own DC_and_CE_loss) on CombinedQuantENet, warm-
started from the ALREADY-CALIBRATED checkpoint (compression/post-
quantization/calibrate_8_2_relu_no_reg_fullwidth.py's own output --
calibration confirmed load-bearing for this architecture family, not
optional). Differs from that sibling script only in: CONTEXT_PATTERN
("dense_dilation", not "dense_dilation_d2_projected"), N_EPOCHS=10, and a
configurable LR (default 5e-3 here, vs the base 0.001 the real HPC job
currently uses -- motivated by this session's own real HPC log for this
run looking stuck/very slow at the default LR).

Usage:
    python compression/qat_smoke_8_2_relu_no_reg_fullwidth.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from brevitas.graph.calibrate import calibration_mode

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enet"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compression" / "hawq"))

from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet, block_names_for, expand_stage_bits  # noqa: E402
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402
from sensitivity import load_real_batches, NNUNET_PREPROCESSED  # noqa: E402

CALIBRATED_CHECKPOINT_PATH = REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c" / \
    "nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_fullwidth_calibrated__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
STAGE_BITS_PATH = REPO_ROOT / "compression" / "hawq" / "stage_bits_8_2_relu_no_reg_fullwidth_minres.json"
DATASET_NAME = "Dataset509_ARCADE_1x1_4c"

STAGE_MODULE_ATTRS = {
    "initial": ("initial",),
    "stage1": ("down1", "regular1"),
    "context": ("down2", "stage2", "stage3"),
    "stage4": ("up4", "regular4"),
    "stage5": ("up5", "regular5", "final"),
}
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
CHANNELS = (4, 16, 32, 16, 4)
CONTEXT_PATTERN = "dense_dilation"
N_CLASSES = 5
N_BATCHES = 25
N_EPOCHS = 10
LR = 5e-3


def load_calibration_batches(dataset_name: str, n_images: int, seed: int = 0) -> list[torch.Tensor]:
    preprocessed_dir = NNUNET_PREPROCESSED / dataset_name / "nnUNetPlans_2d"
    image_files = sorted(p for p in preprocessed_dir.glob("*.npy") if not p.name.endswith("_seg.npy"))
    rng = random.Random(seed)
    sampled = rng.sample(image_files, k=min(n_images, len(image_files)))
    return [torch.from_numpy(np.load(p)).float() for p in sampled]


def pseudo_dice(logits: torch.Tensor, seg: torch.Tensor, n_classes: int) -> list[float]:
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(1, n_classes):
        p, g = (pred == c), (seg.squeeze(1) == c)
        inter = (p & g).sum().float()
        denom = p.sum().float() + g.sum().float()
        dices.append((2 * inter / denom).item() if denom > 0 else 1.0)
    return dices


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage_bits = json.load(open(STAGE_BITS_PATH))
    block_names = block_names_for(BOTTLENECKS_PER_STAGE)
    block_w, block_a = expand_stage_bits(
        stage_bits["stage_weight_bits"], stage_bits["stage_act_bits"], STAGE_MODULE_ATTRS, block_names,
    )
    print(f"Per-stage bits: weights={stage_bits['stage_weight_bits']}, acts={stage_bits['stage_act_bits']}")
    print(f"LR={LR}, N_EPOCHS={N_EPOCHS}")

    model = CombinedQuantENet.from_pretrained(
        CALIBRATED_CHECKPOINT_PATH, block_w, block_a,
        out_channels=N_CLASSES, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False,
        use_strided=True, use_dsc=False, dsc_no_projection=True, separable_dilated=False,
        trainable_slope=False,
    ).to(device)

    loss_fn = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
        weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    batches = load_real_batches(DATASET_NAME, N_BATCHES, seed=0)
    print(f"Loaded {len(batches)} real (image, seg) batches from {DATASET_NAME}. Training {N_EPOCHS} epochs.")

    for epoch in range(N_EPOCHS):
        model.train()
        losses = []
        dices_per_batch = []
        n_used = 0
        for img, seg in batches:
            img, seg = img.to(device), seg.to(device)
            optimizer.zero_grad()
            try:
                out = model(img)
            except RuntimeError as error:
                if "shape mismatch" in str(error) or "size" in str(error).lower():
                    continue
                raise
            loss = loss_fn(out, seg)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            dices_per_batch.append(pseudo_dice(out.detach(), seg, N_CLASSES))
            n_used += 1
        mean_loss = sum(losses) / len(losses)
        mean_dice_per_class = [sum(d[c] for d in dices_per_batch) / len(dices_per_batch) for c in range(N_CLASSES - 1)]
        print(f"Epoch {epoch}: n_batches_used={n_used}/{len(batches)}, mean_loss={mean_loss:.4f}, "
              f"pseudo_dice_per_class={[round(d, 4) for d in mean_dice_per_class]}, "
              f"mean_pseudo_dice={sum(mean_dice_per_class) / len(mean_dice_per_class):.4f}")


if __name__ == "__main__":
    main()
