"""One-off numerical check: does a fixed (non-learned), depthwise
ConvTranspose2d(kernel_size=4, stride=2, padding=1) with the classic
"bilinear" kernel weight reproduce
F.interpolate(x, size=(2H,2W), mode="bilinear", align_corners=False)
EXACTLY (bit-for-bit up to float rounding)?

Used to decide whether up4/up5's main-path upsample (real model:
F.interpolate bilinear, non-FINN-synthesizable Resize op) can be replaced
by a fixed/frozen learned-conv-transpose equivalent instead of a fresh/
random one -- i.e. whether the FINN mirror can be topologically exact
here, the same way down1/down2's shortcut_proj can be made exact via a
fixed padded-identity 1x1 conv.

Run with: C:\\DEV\\Python310\\python.exe hardware\\verify_bilinear_kernel.py
"""
import torch
import torch.nn.functional as F


def bilinear_kernel_2x(channels: int) -> torch.Tensor:
    """Fixed depthwise ConvTranspose2d weight, shape (channels, 1, 4, 4),
    reproducing 2x bilinear upsampling with align_corners=False exactly.

    NOTE: the naive version of this (plain ConvTranspose2d(kernel=4,
    stride=2, padding=1) on the un-padded input) is WRONG at the border --
    align_corners=False's src-position formula clamps to the edge sample
    (replicate semantics) at the boundary, whereas ConvTranspose2d's
    "padding" trims a zero-padded implicit input (zero semantics). The
    correct exact reproduction needs: replicate-pad the input by 1 pixel
    on each side first, then run a stride-2/kernel-4 conv_transpose with
    padding=3 (trims 3 from each side of the (Lp-1)*2+4 raw output).
    Verified bit-exact (max_abs_err ~1e-7, float rounding only) against
    F.interpolate for several non-square/odd shapes below.
    """
    k1d = torch.tensor([1.0, 3.0, 3.0, 1.0]) / 4.0
    k2d = torch.outer(k1d, k1d)  # (4,4), separable bilinear kernel
    w = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return w


def bilinear_upsample_2x_exact(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv_transpose2d(xp, w, bias=None, stride=2, padding=3, groups=channels)


def main():
    torch.manual_seed(0)
    for H, W, C in [(8, 8, 3), (7, 9, 4), (16, 24, 2), (5, 5, 1), (9, 9, 6), (2, 2, 1)]:
        x = torch.randn(1, C, H, W)

        ref = F.interpolate(x, size=(2 * H, 2 * W), mode="bilinear", align_corners=False)

        w = bilinear_kernel_2x(C)
        got = bilinear_upsample_2x_exact(x, w)

        max_abs_err = (ref - got).abs().max().item()
        match = torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
        print(f"H={H} W={W} C={C}: shapes ref={tuple(ref.shape)} got={tuple(got.shape)} "
              f"max_abs_err={max_abs_err:.3e} exact_match={match}")


if __name__ == "__main__":
    main()
