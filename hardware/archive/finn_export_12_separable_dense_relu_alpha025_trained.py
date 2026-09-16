"""Export a FINN-compatible, PER-LAYER HAWQ mirror of the REAL, trained
nnUNetTrainerLayerQuantENet_12_separable_dense_relu_perlayer checkpoint
(alpha=0.25 per-layer HAWQ bit-width assignment), with REAL TRAINED WEIGHTS
transferred wherever the FINN-safe topology is structurally identical to
the real LayerQuantENet (separable_dilated=True).

Byte-for-byte copy of
finn_export_12_dense_relu_warmstart150ep_alpha025_trained.py (the DENSE,
non-separable sibling this mirrors) with only CHECKPOINT/BITS_FILE/
separable_dilated=True changed -- see that file's own module docstring for
the full weight-transfer rationale (regular1/regular4/regular5/stage2/
stage3 reuse the REAL LayerQuantRegularBottleneck class unmodified;
initial/down1/down2/up4/up5's real sub-modules use the exact same site
names as the real blocks; shortcut_proj/main_up/initial.branch_quant have
no checkpoint counterpart and stay at their fixed/frozen or freshly-
constructed values; final.bias transfers through the same generic loop
since bias_quant=Int32Bias only affects forward-time quantization, not the
parameter's shape/semantics) -- all of this applies UNCHANGED to the
separable_dilated=True case; only stage2/stage3's dilated blocks have
different real site names (conv.0/conv.3 instead of a single conv), which
`layer_names_for(separable_dilated=True, ...)` + the real
LayerQuantRegularBottleneck already produce/consume identically on both
sides (checkpoint and FINN mirror), so the generic strict=False name+shape
transfer picks them up with no special-casing needed.

REAL checkpoint: data/nnUNet_results/Dataset509_ARCADE_1x1_4c/
nnUNetTrainerLayerQuantENet_12_separable_dense_relu_perlayer_
12_separable_dense_relu_joint_alpha0.25_perlayer_candidatebits468_
forcedsp_lut70_ft15ep__nnUNetPlans__2d/fold_0/checkpoint_best.pth (the
already-trained NON-warmstart separable sibling -- user confirmed 2026-09-XX
to use this checkpoint rather than first QAT-training a
12_separable_dense_relu_warmstart150ep variant, which was never actually
fine-tuned at this alpha/perlayer recipe).

Usage (run inside the pytorch training container):
    docker exec <container> python /workspace/LightM-UNet/hardware/finn_export_12_separable_dense_relu_alpha025_trained.py

Output: hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_alpha025_trained_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_alpha025_trained_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from brevitas.graph.calibrate import calibration_mode

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.LayerQuantENet import layer_names_for  # noqa: E402
from nnunetv2.nets.LayerQuantEnetFINN import LayerQuantEnetFINN  # noqa: E402
from finn_export_s13_leaky_frozen import export_model  # noqa: E402
from finn_export_12_separable_dense_relu_alpha025_dummy import (  # noqa: E402
    load_layer_bits, CHANNELS, BOTTLENECKS_PER_STAGE, CONTEXT_PATTERN, SEPARABLE_DILATED,
    DEFAULT_BITS_FILE,
)

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerLayerQuantENet_12_separable_dense_relu_perlayer_12_separable_dense_relu_joint_"
      "alpha0.25_perlayer_candidatebits468_forcedsp_lut70_ft15ep__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_PREPROCESSED_DIR = (
    REPO_ROOT / "data" / "nnUNet_preprocessed" / "Dataset509_ARCADE_1x1_4c" / "nnUNetPlans_2d"
)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """Same rationale as the dense sibling's own helper -- nnU-Net
    preprocessed patches have per-case native H/W not guaranteed divisible
    by this network's total downsampling factor of 8."""
    h, w = x.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def load_calibration_images(
    preprocessed_dir: Path, n: int | None = None, split: str = "train",
) -> list[torch.Tensor]:
    """Same real preprocessed image patches as the dense sibling's own
    helper -- see that file's docstring for the full rationale (n=None uses
    the full ~1000-case train split, only a few minutes on CPU)."""
    paths = sorted(preprocessed_dir.glob(f"{split}_*_p0000.npy"))
    if not paths:
        raise FileNotFoundError(f"no {split}_*_p0000.npy patches found under {preprocessed_dir}")
    if n is not None:
        paths = paths[:n]
    images = []
    for path in paths:
        arr = np.load(path)  # (1, 1, H, W) float32, already z-score normalized
        images.append(_pad_to_multiple(torch.from_numpy(arr).float()))
    print(f"  Loaded {len(images)} real calibration images from {preprocessed_dir} (split={split}).")
    return images


def load_real_weights(model: "LayerQuantEnetFINN", checkpoint_path: Path) -> None:
    """Generic strict=False name+shape state-dict transfer -- identical to
    the dense sibling's own (see that file's module docstring)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state_dict = checkpoint["network_weights"]
    model_state_dict = model.state_dict()

    transferable = {
        key: value for key, value in source_state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    assert not unexpected, f"unexpected keys after strict=False load (should be impossible): {unexpected}"
    n_shape_mismatch = sum(
        1 for key, value in source_state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape != value.shape
    )
    print(
        f"load_real_weights({checkpoint_path}): transferred {len(transferable)}/{len(model_state_dict)} "
        f"model keys ({n_shape_mismatch} shape mismatches, {len(missing)} left uninitialized -- expected "
        f"for down1/down2.shortcut_proj + up4/up5.main_up (fixed/frozen, no real counterpart) and "
        f"Brevitas-only quantizer scale params)."
    )
    still_frozen = [k for k in missing if "shortcut_proj" in k or "main_up" in k]
    print(f"  -> {len(still_frozen)}/{len(missing)} of the uninitialized keys are the expected frozen "
          f"shortcut_proj/main_up params (untouched, as intended): {still_frozen}")


def calibrate_runtime_stats(model: "LayerQuantEnetFINN", calibration_images: list[torch.Tensor]) -> None:
    """Same runtime-stats calibration fix as the dense sibling's own (see
    that file's docstring for the full RuntimeError/root-cause derivation)."""
    with torch.no_grad(), calibration_mode(model):
        for image in calibration_images:
            model(image)
    print(f"  Calibrated runtime-stats scaling buffers over {len(calibration_images)} real training patches.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument(
        "--calibration-images", type=int, default=-1,
        help="number of real training patches to calibrate on; -1 (default) uses the full train split",
    )
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    shape_kwargs = dict(
        out_channels=args.out_channels, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
        context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False,
        separable_dilated=SEPARABLE_DILATED,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building REAL-weight, per-layer-bit-width FINN-safe 12_separable_dense_relu "
          f"(alpha=0.25) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
        channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern=CONTEXT_PATTERN,
        separable_dilated=SEPARABLE_DILATED,
    ).eval()

    print("\n=== Loading real trained checkpoint ===")
    load_real_weights(model, checkpoint_path)

    print("\n=== Calibrating runtime-stats activation scales on real training data ===")
    n_calib = None if args.calibration_images < 0 else args.calibration_images
    calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
    calibrate_runtime_stats(model, calibration_images)

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_12_separable_dense_relu_alpha025_trained_int8"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
