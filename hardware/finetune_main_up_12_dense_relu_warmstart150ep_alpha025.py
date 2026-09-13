"""Fine-tune ONLY `up4.main_up`/`up5.main_up` (the frozen bilinear-kernel
ConvTranspose2d substitute, see finn_export_12_dense_relu_warmstart150ep_
alpha025_dummy.py's `_bilinear_kernel_conv_transpose`) so its weights drift
away from the exact bilinear kernel's mostly-zero, block-diagonal
structure, while every other parameter in the network stays byte-identical
to the real trained checkpoint.

Motivation: the exact frozen bilinear kernel is suspected (not yet
confirmed against FINN source) to trigger a dangling-tensor wiring bug
inside FINN's `step_enet_convert_to_hw` -- see
memories/repo/finn_12_dense_relu_alpha025_perlayer.md "Bug #3". A fresh,
ordinary *learned* ConvTranspose2d (as used by the sibling
12_separable_dense_relu_min4 family's FINN mirror) sidesteps this, but
would be numerically unrelated to the real trained model at that site
(nothing to transfer -- the real op there is F.interpolate, not a learned
conv). This script gets the best of both: start `main_up`'s weight at the
EXACT bilinear kernel (mathematically identical to the real op on interior
pixels), then fine-tune it with a feature-matching (distillation) loss
against the real bilinear-interpolate output computed on the SAME
(quantized) input tensor, with every other parameter frozen. This:
  - needs no segmentation labels (self-supervised on the calibration image
    set alone),
  - starts from the exact solution, so should stay very close to bit-exact
    even after a few fine-tuning steps,
  - is expected (not guaranteed) to perturb the weight matrix away from
    the pathological all-zero off-diagonal pattern enough to dodge Bug #3.

Small/fast enough to run locally on CPU (~0.2s/image forward+backward,
only 2 tiny conv kernels have require_grad=True) -- no HPC/Slurm needed.

Usage (inside the pytorch training container):
    docker exec <container> python /workspace/LightM-UNet/hardware/finetune_main_up_12_dense_relu_warmstart150ep_alpha025.py

Output: hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_ftmainup_int8.onnx
        hardware/outputs/finn_exports/quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_ftmainup_int8_final_bias.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from finn_export_s13_leaky_frozen import export_model
from finn_export_12_dense_relu_warmstart150ep_alpha025_dummy import (
    LayerQuantEnetFINN, load_layer_bits, layer_names_for, CHANNELS, BOTTLENECKS_PER_STAGE, CONTEXT_PATTERN,
    OUT_DIR, DEFAULT_BITS_FILE,
)
from finn_export_12_dense_relu_warmstart150ep_alpha025_trained import (
    load_real_weights, load_calibration_images, calibrate_runtime_stats,
    DEFAULT_CHECKPOINT, DEFAULT_PREPROCESSED_DIR,
)


def _as_plain_tensor(x):
    """Brevitas QuantTensor (return_quant_tensor=True) or plain Tensor -> plain Tensor."""
    return x.value if hasattr(x, "value") else x


def finetune_main_up(
    model: "LayerQuantEnetFINN", finetune_images: list[torch.Tensor], epochs: int, lr: float,
) -> None:
    """Freezes every parameter except up4.main_up.weight/up5.main_up.weight,
    then fits those 2 kernels to match real F.interpolate(bilinear) on the
    same (quantized) input each sees at its real position in the network,
    via forward hooks capturing main_act's output (= main_up's real input)
    and main_up's own output on every training forward pass."""
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for stage_name in ("up4", "up5"):
        w = getattr(model, stage_name).main_up.weight
        w.requires_grad_(True)
        trainable.append(w)

    captured: dict[str, torch.Tensor] = {}

    def _save(key):
        def _hook(_module, _inputs, output):
            captured[key] = output
        return _hook

    handles = []
    for stage_name in ("up4", "up5"):
        block = getattr(model, stage_name)
        handles.append(block.main_act.register_forward_hook(_save(f"{stage_name}_in")))
        handles.append(block.main_up.register_forward_hook(_save(f"{stage_name}_out")))

    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.eval()  # BN/dropout stay in inference mode; autograd tracking is unaffected.

    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for image in finetune_images:
                optimizer.zero_grad()
                model(image)
                loss = torch.zeros(())
                for stage_name in ("up4", "up5"):
                    quant_in = _as_plain_tensor(captured[f"{stage_name}_in"])
                    pred = _as_plain_tensor(captured[f"{stage_name}_out"])
                    with torch.no_grad():
                        reference = F.interpolate(
                            quant_in, scale_factor=2, mode="bilinear", align_corners=False,
                        )
                    loss = loss + F.mse_loss(pred, reference)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"  epoch {epoch + 1}/{epochs}: mean loss = {epoch_loss / len(finetune_images):.6e}")
    finally:
        for h in handles:
            h.remove()

    for w in trainable:
        w.requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits-file", default=str(DEFAULT_BITS_FILE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--preprocessed-dir", default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--calibration-images", type=int, default=-1)
    parser.add_argument("--finetune-images", type=int, default=200,
                         help="number of real training patches used for the main_up fine-tune loop")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    print(f"\n=== Building REAL-weight FINN-safe 12_dense_relu_warmstart150ep (alpha=0.25) ===")
    model = LayerQuantEnetFINN(
        layer_weight_bits, layer_act_bits, in_channels=args.in_channels, out_channels=args.out_channels,
    ).eval()

    print("\n=== Loading real trained checkpoint ===")
    real_final_bias = load_real_weights(model, checkpoint_path)

    print("\n=== Calibrating runtime-stats activation scales on real training data ===")
    n_calib = None if args.calibration_images < 0 else args.calibration_images
    calibration_images = load_calibration_images(Path(args.preprocessed_dir), n=n_calib)
    calibrate_runtime_stats(model, calibration_images)

    print(f"\n=== Fine-tuning main_up (up4/up5) only, {args.epochs} epochs, lr={args.lr} ===")
    finetune_images = calibration_images[: args.finetune_images]
    finetune_main_up(model, finetune_images, epochs=args.epochs, lr=args.lr)

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    name = "quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_ftmainup_int8"
    export_model(model, name, dummy)

    bias_path = OUT_DIR / f"{name}_final_bias.npy"
    np.save(bias_path, real_final_bias.detach().numpy())
    print(f"  Saved real final.bias side-car: {bias_path}")

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
