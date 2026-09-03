"""MobileNetV2 (Sandler et al., 2018, CVPR), adapted into an encoder-decoder
segmentation network for this repo's own nnU-Net pipeline.

Architecturally verified against the official TensorFlow-slim source
(models/research/slim/nets/mobilenet/mobilenet_v2.py, cloned into this
repo's own models/ directory -- V2_DEF's own `spec` list read directly, not
reconstructed from memory of the paper text alone). Confirmed real
(t, c, n, s) inverted-residual schedule (t=expand ratio, c=output channels,
n=block repeat count, s=stride of the FIRST block in the group, every
other repeat stride=1):
    stem: Conv2d(in, 32, k=3, s=2) -> BN -> ReLU6
    t=1, c=16,  n=1, s=1
    t=6, c=24,  n=2, s=2
    t=6, c=32,  n=3, s=2
    t=6, c=64,  n=4, s=2
    t=6, c=96,  n=3, s=1   # no downsample -- channel change only
    t=6, c=160, n=3, s=2
    t=6, c=320, n=1, s=1
    head: Conv2d(320, 1280, k=1) -> BN -> ReLU6
5 stride-2 points total (stem + the 4 stride-2 stage-starts) -- REAL
paper/backbone depth (/32 total downsampling), deliberately NOT truncated
to this repo's usual /8 (3-downsample) convention every ENet/ERFNet variant
here uses (explicit choice, see this file's own git history/conversation
context -- full paper fidelity over cross-architecture downsample-depth
comparability).

Inverted residual block (real MobileNetV2 block, confirmed against the
source's own `expanded_conv` op + this repo's cross-check against
torchvision's own well-established MobileNetV2 implementation, same block
definition): 1x1 expand (BN+ReLU6, SKIPPED when expand_ratio==1 -- the
first stage's own t=1 blocks have no expand conv at all, matching
`expansion_size=expand_input(1, divisible_by=1)` in the source) -> 3x3
depthwise (stride s, BN+ReLU6) -> 1x1 project (BN, LINEAR -- no activation,
the paper's own "linear bottleneck" this architecture is named for) ->
residual add IFF stride==1 AND in_channels==out_channels (matches the
source's own `residual=True` default plus the standard shape-compatibility
guard every implementation of this block uses).

Decoder is this repo's OWN addition (MobileNetV2 is a classification
backbone in the source, no decoder defined by the paper) -- NOT specified
by any external spec, so no "verified against the source" claim is made
for it. Mirrors the 5 encoder downsample points with 5 plain upsample
(ConvTranspose2d stride 2) + BN + ReLU6 stages (same activation convention
as the encoder, no skip connections -- same "small decoder, no long-range
skips" philosophy ENet.py/ERFNet.py both already use in this repo),
channel schedule following the encoder's own real channel scale at
corresponding depths (1280->160->96->32->24->out_channels), ending in a
bare Conv2d 1x1 projection to logits (matches ENet.py's/ERFNet.py's own
"final" layer convention -- no BN/activation on the raw logits layer).

width_mult: the REAL paper's own width-scaling knob (not this repo's usual
5-value `channels` convention, which doesn't map cleanly onto MobileNetV2's
own finer-grained 8-stage channel schedule) -- every channel count above is
scaled by width_mult and rounded to the nearest multiple of 8 (divisible_by
=8, matching the source's own default rounding rule), same convention the
source's own mobilenet_v2_140/mobilenet_v2_050/mobilenet_v2_035 wrappers
use. Default 1.0 = the real base "MobileNetV2 1.0" architecture.
"""
from __future__ import annotations

import torch
from torch import nn


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Same rounding rule the reference implementation uses throughout
    (round to nearest multiple of `divisor`, never round down by more than
    10%) -- ensures every channel count stays hardware-friendly under an
    arbitrary width_mult, not just the paper's own named multipliers."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """1x1 expand (skipped iff expand_ratio==1) -> 3x3 depthwise (stride s)
    -> 1x1 project (linear, no activation) -> residual iff stride==1 and
    in_channels==out_channels. See this file's own module docstring for
    the full verification/rationale."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: float):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"InvertedResidual stride must be 1 or 2, got {stride}.")
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
        layers += [
            ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_residual else self.conv(x)


class UpsampleBlock(nn.Sequential):
    """Decoder's own upsample-by-2 primitive -- plain ConvTranspose2d, same
    ReLU6 convention the encoder uses throughout (not ERFNet.py's plain
    ReLU, matching MobileNetV2's own activation choice for consistency
    within this one file instead)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


# Real (t, c, n, s) schedule -- see module docstring.
_INVERTED_RESIDUAL_SETTING: tuple[tuple[int, int, int, int], ...] = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 20,
        width_mult: float = 1.0,
        stem_channels: int = 32,
        head_channels: int = 1280,
    ):
        super().__init__()
        stem_c = _make_divisible(stem_channels * width_mult)
        head_c = _make_divisible(head_channels * max(1.0, width_mult))

        # -- Encoder (real architecture, /32 total downsample) --------------
        self.stem = ConvBNReLU6(in_channels, stem_c, kernel_size=3, stride=2)

        self.stages = nn.ModuleList()
        in_c = stem_c
        stage_out_channels = []  # per-stage output width, for the decoder's own channel schedule
        for t, c, n, s in _INVERTED_RESIDUAL_SETTING:
            out_c = _make_divisible(c * width_mult)
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_c, out_c, stride, expand_ratio=t))
                in_c = out_c
            self.stages.append(nn.Sequential(*blocks))
            stage_out_channels.append(out_c)
        self.head = ConvBNReLU6(in_c, head_c, kernel_size=1)

        # -- Decoder (this repo's own addition -- see module docstring) -----
        # Channel schedule mirrors the encoder's own real scale at the 5
        # downsample depths: head_c -> stage[5] (c=160) -> stage[3] (c=64)
        # -> stage[1] (c=24) -> stem_c -> out_channels. Indices below are
        # into stage_out_channels (0-based over the 7 (t,c,n,s) rows), at
        # the rows that immediately FOLLOW each of the 4 non-stem stride-2
        # points (rows 1, 2, 3, 5 -- 0-indexed).
        d0, d1, d2, d3 = (stage_out_channels[i] for i in (5, 3, 1, 0))
        self.up1 = UpsampleBlock(head_c, d0)
        self.up2 = UpsampleBlock(d0, d1)
        self.up3 = UpsampleBlock(d1, d2)
        self.up4 = UpsampleBlock(d2, d3)
        self.up5 = UpsampleBlock(d3, stem_c)
        self.final = nn.Conv2d(stem_c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.final(x)


if __name__ == "__main__":
    dummy = torch.randn(1, 1, 512, 512)
    net = MobileNetV2(in_channels=1, out_channels=5).eval()
    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (1, 5, 512, 512), f"got {tuple(out.shape)}"
    assert len(net.stages) == 7, f"expected 7 stages, got {len(net.stages)}"
    expected_repeats = [1, 2, 3, 4, 3, 3, 1]
    for i, (stage, expected_n) in enumerate(zip(net.stages, expected_repeats)):
        assert len(stage) == expected_n, f"stage {i}: expected {expected_n} blocks, got {len(stage)}"
    # First stage (t=1) blocks must have NO expand conv (conv[0] is
    # directly the depthwise ConvBNReLU6, not a 1x1 expand).
    assert len(net.stages[0][0].conv) == 3, "stage0 block0 (t=1) should skip the expand conv (3 modules, not 4)"
    assert net.stages[0][0].conv[0][0].kernel_size == (3, 3), "stage0 block0 (t=1) conv[0] should be the depthwise 3x3 directly"
    # A later stage's first block (t=6) DOES have the expand conv.
    assert len(net.stages[1][0].conv) == 4, "stage1 block0 (t=6) should have the expand conv (4 modules)"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"MobileNetV2 self-test PASSED: builds, forward-passes to {tuple(out.shape)}, "
          f"7-stage (t,c,n,s) schedule verified (repeats {expected_repeats}), "
          f"t=1 expand-skip verified, t=6 expand-present verified. {n_params} params.")
