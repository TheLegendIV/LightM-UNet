from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.nets.ENet import (
    DownsamplingBottleneck,
    InitialBlock,
    RegularBottleneck,
    UpsamplingBottleneck,
)


def _context_stage(channels: int, dropout_p: float = 0.1) -> list[RegularBottleneck]:
    """Original ENet 8-block dilated context pattern."""
    return [
        RegularBottleneck(channels, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=2, dilation=2, dropout_p=dropout_p),
        RegularBottleneck(channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=4, dilation=4, dropout_p=dropout_p),
        RegularBottleneck(channels, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=8, dilation=8, dropout_p=dropout_p),
        RegularBottleneck(channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=16, dilation=16, dropout_p=dropout_p),
    ]


def _extra_blocks(channels: int, dropout_p: float = 0.1) -> list[RegularBottleneck]:
    """4 extra context blocks appended per stage: regular, dil8, asym, dil4."""
    return [
        RegularBottleneck(channels, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=8, dilation=8, dropout_p=dropout_p),
        RegularBottleneck(channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=4, dilation=4, dropout_p=dropout_p),
    ]


def _asym_extra_blocks(channels: int, dropout_p: float = 0.1) -> list[RegularBottleneck]:
    """Asymmetric-heavy extra blocks: reg, asym, asym, dil4 (vs reg, dil8, asym, dil4).
    Replaces the dil8 block with a second asymmetric block for vessel-oriented bias."""
    return [
        RegularBottleneck(channels, dropout_p=dropout_p),
        RegularBottleneck(channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=dropout_p),
        RegularBottleneck(channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=dropout_p),
        RegularBottleneck(channels, padding=4, dilation=4, dropout_p=dropout_p),
    ]


class ENetUpscaled(nn.Module):
    """ENet with five architectural upscaling changes.

    Changes vs ENetOriginal:
    - [C5] 2 extra regular bottlenecks after InitialBlock at H/2
    - [C3] Downsampling delayed: stage 2 runs flat at H/4 (no stride), stride moves to stage 3 entry
    - [C2] Stage 3 gets 4 extra blocks (total 12); new stage 4 gets 4 blocks
    - [C1] New encoder stage 4 at H/16 with ch4 channels
    - [C6] Matching decoder stage for H/16→H/8; decoder uses bilinear interpolation (vs MaxUnpool)

    channels: 8-tuple (ch0, ch1, ch2, ch3, ch4, ch_d3, ch_d2, ch_d1)
        ch0   = initial block output  (H/2)
        ch1   = stage 1 encoder       (H/4)
        ch2   = stage 2 flat encoder  (H/4, no stride)
        ch3   = stage 3 encoder       (H/8)
        ch4   = stage 4 deep encoder  (H/16)
        ch_d3 = decoder level H/8
        ch_d2 = decoder level H/4
        ch_d1 = decoder level H/2  (before final ConvTranspose)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: Sequence[int] = (20, 72, 72, 144, 256, 144, 72, 20),
        asym_heavy_stage3: bool = False,
    ):
        super().__init__()
        if len(channels) != 8:
            raise ValueError(
                "ENetUpscaled expects an 8-element channels tuple: "
                "(ch0, ch1, ch2, ch3, ch4, ch_d3, ch_d2, ch_d1)."
            )
        if any(c % 4 != 0 for c in channels):
            raise ValueError("All ENetUpscaled channels must be divisible by 4.")
        ch0, ch1, ch2, ch3, ch4, ch_d3, ch_d2, ch_d1 = (int(c) for c in channels)
        if ch0 <= in_channels:
            raise ValueError("ch0 must exceed in_channels.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(int(c) for c in channels)

        # --- Encoder ---

        # [C5] Initial block + 2 extra regular bottlenecks at H/2
        self.initial = InitialBlock(in_channels, ch0)
        self.initial_extra = nn.Sequential(
            RegularBottleneck(ch0, dropout_p=0.01),
            RegularBottleneck(ch0, dropout_p=0.01),
        )

        # Stage 1: stride-2 downsample to H/4, then 4 regular blocks
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])

        # [C3] Stage 2: flat at H/4 (NO downsampling), optional channel projection
        self.proj1_to_2 = (
            nn.Identity()
            if ch2 == ch1
            else nn.Sequential(
                nn.Conv2d(ch1, ch2, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch2),
                nn.PReLU(ch2),
            )
        )
        self.stage2_flat = nn.Sequential(*_context_stage(ch2))

        # [C3] Stage 3: downsampling now happens here (delayed from stage 2)
        # [C2] 12 blocks = 8 original context + 4 extra
        self.down2 = DownsamplingBottleneck(ch2, ch3, dropout_p=0.1)
        extra_fn = _asym_extra_blocks if asym_heavy_stage3 else _extra_blocks
        self.stage3 = nn.Sequential(*_context_stage(ch3), *extra_fn(ch3))

        # [C1/C6] New stage 4: additional downsample to H/16 + 4 context blocks
        self.down3 = DownsamplingBottleneck(ch3, ch4, dropout_p=0.1)
        self.stage4 = nn.Sequential(*_extra_blocks(ch4))

        # --- Decoder (bilinear interpolation throughout for channel flexibility) ---

        # [C6] H/16 → H/8
        self.up_deep = UpsamplingBottleneck(ch4, ch_d3)
        self.dec3 = nn.Sequential(
            RegularBottleneck(ch_d3, dropout_p=0.1, relu=True),
            RegularBottleneck(ch_d3, dropout_p=0.1, relu=True),
        )

        # H/8 → H/4
        self.up_mid = UpsamplingBottleneck(ch_d3, ch_d2)
        self.dec2 = nn.Sequential(
            RegularBottleneck(ch_d2, dropout_p=0.1, relu=True),
            RegularBottleneck(ch_d2, dropout_p=0.1, relu=True),
        )

        # H/4 → H/2
        self.up_shal = UpsamplingBottleneck(ch_d2, ch_d1)
        self.dec1 = RegularBottleneck(ch_d1, dropout_p=0.1, relu=True)

        # H/2 → H
        self.final = nn.ConvTranspose2d(ch_d1, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.initial(x)
        x = self.initial_extra(x)

        x, _, sz1 = self.down1(x)   # H/2 → H/4
        x = self.stage1(x)

        x = self.proj1_to_2(x)
        x = self.stage2_flat(x)     # flat at H/4

        x, _, sz2 = self.down2(x)   # H/4 → H/8
        x = self.stage3(x)

        x, _, sz3 = self.down3(x)   # H/8 → H/16
        x = self.stage4(x)

        # Decoder (bilinear, indices discarded)
        x = self.up_deep(x, sz3)    # H/16 → H/8
        x = self.dec3(x)

        x = self.up_mid(x, sz2)     # H/8 → H/4
        x = self.dec2(x)

        x = self.up_shal(x, sz1)    # H/4 → H/2
        x = self.dec1(x)

        x = self.final(x)           # H/2 → H
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
