"""Export a FINN-compatible, PER-LAYER HAWQ mirror of the REAL, QAT
fine-tuned nnUNetTrainerENet_12_dense_relu_warmstart150ep checkpoint
(alpha=0.25 per-layer HAWQ bit-width assignment), with REAL TRAINED WEIGHTS
transferred wherever the FINN-safe topology is structurally identical to
the real LayerQuantENet.

Uses the same enet/nnunetv2/nets/LayerQuantEnetFINN.py architecture (see
that file's module docstring for the full derivation of the 4 FINN-specific
substitutions) as finn_export_12_dense_relu_warmstart150ep_alpha025_dummy.py
-- only this file's checkpoint loading + weight transfer + final.bias
handling are new.

WEIGHT TRANSFER: since regular1/regular4/regular5/stage2/stage3 reuse the
REAL LayerQuantRegularBottleneck class unmodified, and initial's
conv/pool/bn/act and down1/down2/up4/up5's reduce/conv/expand/up/main_proj/
residual_add/out_act submodules use the exact same site names as the real
LayerQuantInitialBlock/LayerQuantDownsamplingBottleneck/
LayerQuantUpsamplingBottleneck, a single generic strict=False name+shape
state-dict transfer (identical in spirit to LayerQuantENet.from_pretrained's
own -- see enet/nnunetv2/nets/LayerQuantENet.py) transfers EVERY real
trained parameter automatically:
  - shortcut_proj (down1/down2), main_up (up4/up5), and initial.branch_quant
    have no counterpart key in the real checkpoint at all (real model has no
    such submodules), so they are simply never touched by the transfer, and
    are left at their fixed/frozen or freshly-constructed values (shortcut_
    proj/main_up are mathematically EXACT fixed ops, see LayerQuantEnetFINN.py's
    docstring point 1/2; branch_quant is a genuinely NEW rounding point --
    see LayerQuantEnetFINN.py's docstring point 0 -- and needs calibration below
    like any other freshly-added quantizer).
  - every other submodule (initial.conv/pool/bn/act, regular1-5.*,
    stage2/3.*, down1/2's reduce/conv/expand, up4/5's reduce/up/expand/
    main_proj) has an identical key+shape match and transfers in full.
  - `final.bias` also has an identical key+shape match: bias_quant=Int32Bias
    (see dummy script's docstring point 3) quantizes the bias at forward
    time using `final`'s own input/weight scales, but the underlying
    `.bias` parameter itself stays a plain real-valued (out_channels,)
    tensor, byte-for-byte the same shape/semantics as the real checkpoint's
    `final.bias` -- so it transfers via the SAME generic loop, no special-
    casing needed.

REAL checkpoint: data/nnUNet_results/Dataset509_ARCADE_1x1_4c/
nnUNetTrainerLayerQuantENet_12_dense_relu_warmstart150ep_perlayer_
12_dense_relu_warmstart150ep_joint_alpha0.25_perlayer_candidatebits468_
forcedsp_lut70_ft15ep__nnUNetPlans__2d/fold_0/checkpoint_best.pth.

Usage (run inside the pytorch training container):
    docker exec <container> python /workspace/LightM-UNet/hardware/finn_export_12_dense_relu_warmstart150ep_alpha025_trained.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8.onnx \\
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
from finn_export_12_dense_relu_warmstart150ep_alpha025_dummy import (  # noqa: E402
    load_layer_bits, CHANNELS, BOTTLENECKS_PER_STAGE, CONTEXT_PATTERN,
    DEFAULT_BITS_FILE,
)

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerLayerQuantENet_12_dense_relu_warmstart150ep_perlayer_12_dense_relu_warmstart150ep_joint_"
      "alpha0.25_perlayer_candidatebits468_forcedsp_lut70_ft15ep__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)
DEFAULT_PREPROCESSED_DIR = (
    REPO_ROOT / "data" / "nnUNet_preprocessed" / "Dataset509_ARCADE_1x1_4c" / "nnUNetPlans_2d"
)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """nnU-Net preprocessed patches have per-case native H/W (confirmed:
    e.g. train_101 is (512,511), train_103 is (509,509)) -- not guaranteed
    divisible by this network's total downsampling factor of 8. Reflect-pads
    up to the next multiple on the bottom/right only (pad amount is always
    < 8, well within reflect's own size limit)."""
    h, w = x.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def load_calibration_images(
    preprocessed_dir: Path, n: int | None = None, split: str = "train",
) -> list[torch.Tensor]:
    """Real preprocessed image patches (already normalized by nnU-Net's own
    preprocessing -- see the (1,1,H,W) float32 arrays under
    nnUNetPlans_2d/<split>_*_p0000.npy, NOT the sibling *_seg.npy label
    maps) for calibrating runtime-stats activation scales on representative
    data instead of random noise -- needed since this model is also used
    for accuracy benchmarking, where the calibrated scale value (unlike
    bit-width) does matter. n=None uses the FULL split (~1000 train cases;
    ~0.2s/image forward on CPU at 512x512, so the full set is only a few
    minutes -- more patches give more robust percentile-based scale
    estimates, no downside besides that compute time)."""
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
    """Generic strict=False name+shape state-dict transfer -- identical in
    spirit to LayerQuantENet.from_pretrained's own (see module docstring).
    `final.bias` transfers through this same generic loop (bias_quant=
    Int32Bias only affects the quantization applied at forward time, not
    the underlying parameter's shape/semantics), so no special-casing is
    needed."""
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
    """Root-cause fix for `RuntimeError: Scaling factors are different` in
    down1/down2's residual_add (QuantEltwiseAdd) during the real-weight
    forward pass -- see memories/repo/finn_12_dense_relu_alpha025_perlayer.md
    for the full derivation.

    act_bits sites throughout this model use Brevitas's
    ParamFromRuntimePercentileScaling (confirmed via Int8ActPerTensorFloat's
    MRO), whose actual scale-tracking state is held in NON-PERSISTENT
    buffers (invisible to state_dict, confirmed empirically: state_dict()
    size is identical before/after a forward pass, and a fresh model has
    zero keys under e.g. "down1.residual_add" both before and after
    warm-up). This means: (1) the real checkpoint's own leftover
    "...scaling_impl.value" keys are DEAD/orphaned (from a since-changed
    quantizer config) and are correctly never transferred by
    load_real_weights, and (2) every quant proxy's actual scale is whatever
    its own non-persistent runtime-stats buffer happens to converge to,
    starting from ITS OWN construction-time state -- which, on a freshly
    built LayerQuantEnetFINN, is NOT guaranteed to already agree between two
    operands (e.g. residual_add's `main`/`out`) that are fed by two
    DIFFERENT quant-proxy instances, until each has been calibrated on
    representative data. The real (unmodified) LayerQuantENet class
    coincidentally often "gets away with" a single uncalibrated forward
    pass (still relying on each instance's own from-construction default),
    but this is not something to depend on -- the correct, standard
    Brevitas deployment step is to run several forward passes wrapped in
    `calibration_mode(model)` (from brevitas.graph.calibrate) BEFORE relying
    on the model for real inference/export, letting every runtime-stats
    scaling buffer observe actual data and settle.

    Uses REAL preprocessed nnU-Net training patches (see
    load_calibration_images) rather than random noise: this model is also
    used for accuracy benchmarking, where the calibrated scale value
    (unlike bit-width, which fixes synthesis results either way) directly
    affects clipping/rounding error on real inputs."""
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
        use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=False,
    )
    weight_names, act_names = layer_names_for(**shape_kwargs)
    layer_weight_bits, layer_act_bits = load_layer_bits(Path(args.bits_file), weight_names, act_names)

    print(f"\n=== Building REAL-weight, per-layer-bit-width FINN-safe 12_dense_relu_warmstart150ep "
          f"(alpha=0.25) -- {len(weight_names)} weight sites, {len(act_names)} act sites ===")
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
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

    name = "quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
