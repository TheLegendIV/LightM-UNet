"""Test bench: does nearest-neighbor upsample + a FIXED (exact bilinear-
equivalent) depthwise conv reproduce the real up4/up5 main-branch bilinear
resize, on REAL cached activations from 64 real training samples of the
trained S12 warmstart (alpha0.25) checkpoint?

Motivation: F.interpolate(mode="bilinear") (the real model's up4/up5 main-
branch resize) has no FINN hardware lowering at all (`InferUpsample` hard-
asserts mode=="nearest" only, confirmed against FINN source -- see
memories/repo/finn_gotchas.md). This measures, on REAL feature-map
statistics (not synthetic random tensors like verify_bilinear_kernel.py),
how closely a FINN-legal substitute -- nearest-neighbor upsample + a
depthwise conv FIXED to the exact bilinear-equivalent kernel -- matches the
real ("authentic", non-FINN-buildable) bilinear output.

Derivation of the exact kernel: 2x bilinear upsampling (align_corners=False)
of a nearest-duplicated signal y (y[2k]=y[2k+1]=x[k]) equals a 1D [1,2,1]/4
convolution (stride 1, "same"/padding=1) of y -- solved by matching
coefficients against F.interpolate's own even/odd source-position weights
(0.25/0.75 and 0.75/0.25 respectively). Separable 2D kernel:
outer([1,2,1],[1,2,1])/16. Verified against F.interpolate on synthetic
tensors: interior max abs err ~1e-7 (pure float rounding); the only real
discrepancy is at the border (zero-pad vs align_corners=False's edge-
replicate semantics -- same boundary mismatch `main_up`'s ConvTranspose2d
trick has, see hardware/verify_bilinear_kernel.py).

Usage (inside the pytorch training container):
    docker exec <container> python /workspace/LightM-UNet/hardware/testbench_bilinear_vs_nearest_depthwise_up4up5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunetv2.nets.LayerQuantENet import LayerQuantENet, layer_names_for  # noqa: E402
from finn_export_12_dense_relu_warmstart150ep_alpha025_dummy import (  # noqa: E402
    load_layer_bits, CHANNELS, BOTTLENECKS_PER_STAGE, CONTEXT_PATTERN, DEFAULT_BITS_FILE,
)
from finn_export_12_dense_relu_warmstart150ep_alpha025_trained import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_PREPROCESSED_DIR, load_calibration_images,
)

N_SAMPLES = 64
SHAPE_KWARGS = dict(
    out_channels=5, channels=CHANNELS, bottlenecks_per_stage=BOTTLENECKS_PER_STAGE,
    context_pattern=CONTEXT_PATTERN, use_dilated=True, use_asymmetric=False, use_strided=True,
    use_dsc=False, dsc_no_projection=False, dsc_no_projection_context_only=False, separable_dilated=True,
)


def _as_plain_tensor(x):
    """Brevitas QuantTensor (return_quant_tensor=True) or plain Tensor -> plain Tensor."""
    return x.value if hasattr(x, "value") else x


def _tent_kernel(channels: int) -> torch.Tensor:
    """Fixed depthwise kernel reproducing 2x bilinear upsampling exactly
    (interior only, see module docstring derivation) when applied to a
    nearest-upsampled signal."""
    k1d = torch.tensor([1.0, 2.0, 1.0]) / 4.0
    k2d = torch.outer(k1d, k1d)
    return k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def nearest_plus_depthwise(main: torch.Tensor) -> torch.Tensor:
    channels = main.shape[1]
    y = F.interpolate(main, scale_factor=2, mode="nearest")
    return F.conv2d(y, _tent_kernel(channels), padding=1, groups=channels)


def build_model() -> LayerQuantENet:
    weight_names, act_names = layer_names_for(**SHAPE_KWARGS)
    layer_weight_bits, layer_act_bits = load_layer_bits(DEFAULT_BITS_FILE, weight_names, act_names)
    model = LayerQuantENet.from_pretrained(DEFAULT_CHECKPOINT, layer_weight_bits, layer_act_bits, **SHAPE_KWARGS)
    model.eval()
    return model


def make_hook(name: str, cached: dict):
    def hook(module, args, _output):
        x, output_size = args[0], args[1]
        indices = args[2] if len(args) > 2 else None
        assert indices is None, f"{name}: expected indices=None (upsample_conv decoder), got non-None"
        with torch.no_grad():
            main = module.main_proj(x)
            authentic = _as_plain_tensor(F.interpolate(main, size=output_size[2:], mode="bilinear", align_corners=False))
            approx = nearest_plus_depthwise(_as_plain_tensor(main))
        cached[name].append((authentic.detach().clone(), approx.detach().clone()))
    return hook


def main():
    torch.manual_seed(0)
    print("Building real LayerQuantENet + loading trained checkpoint...")
    model = build_model()
    images = load_calibration_images(DEFAULT_PREPROCESSED_DIR, n=N_SAMPLES)

    cached = {"up4": [], "up5": []}
    handles = [
        model.up4.register_forward_hook(make_hook("up4", cached)),
        model.up5.register_forward_hook(make_hook("up5", cached)),
    ]

    with torch.no_grad():
        for i, img in enumerate(images):
            model(img)
            print(f"  sample {i + 1}/{len(images)} cached.")

    for h in handles:
        h.remove()

    for name in ("up4", "up5"):
        full_err, interior_err, rel_err = [], [], []
        for authentic, approx in cached[name]:
            err = (authentic - approx).abs()
            full_err.append(err.max().item())
            interior_err.append(err[..., 1:-1, 1:-1].max().item())
            rel_err.append((err.mean() / authentic.abs().mean().clamp_min(1e-8)).item())
        full_t, interior_t, rel_t = torch.tensor(full_err), torch.tensor(interior_err), torch.tensor(rel_err)
        print(f"\n=== {name} ({len(cached[name])} samples, real activations) ===")
        print(f"  full-frame max_abs_err:    mean={full_t.mean():.4e}  max={full_t.max():.4e}")
        print(f"  interior-only max_abs_err: mean={interior_t.mean():.4e}  max={interior_t.max():.4e}")
        print(f"  mean_abs_err / |authentic|_mean: mean={rel_t.mean() * 100:.3f}%  max={rel_t.max() * 100:.3f}%")


if __name__ == "__main__":
    main()
