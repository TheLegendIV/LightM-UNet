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
    """4 extra context blocks (+50%): reg, dil8, asym, dil4."""
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
    """ENet with five architectural upscaling changes vs ENetOriginal.

    Changes:
    - [C5] 2 extra regular bottlenecks after InitialBlock (H/2)
    - [C2] +50% blocks in every encoder stage: stage1 4→6, stage2_flat 8→12, stage3 8→12
    - [C3] Downsampling delayed: stage2 runs flat at H/4 (no stride), stride moved to stage3 entry
    - [C1] New encoder stage4 at H/16
    - [C6] Matching decoder stage H/16→H/8; MaxUnpool throughout (paper-faithful)

    Decoder channels are determined by MaxUnpool constraints (must equal encoder input
    of the paired downsampling block): H/8→ch3, H/4→ch2, H/2→ch0.

    channels: 5-tuple (ch0, ch1, ch2, ch3, ch4)
        ch0   = initial block output  (H/2)
        ch1   = stage1 encoder        (H/4, 6 blocks)
        ch2   = stage2 flat encoder   (H/4, 12 blocks, no stride)
        ch3   = stage3 encoder        (H/8, 12 blocks)
        ch4   = stage4 deep encoder   (H/16, 4 blocks)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: Sequence[int] = (20, 72, 72, 144, 256),
        asym_heavy_stage3: bool = False,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(
                "ENetUpscaled expects a 5-element channels tuple: (ch0, ch1, ch2, ch3, ch4)."
            )
        if any(c % 4 != 0 for c in channels):
            raise ValueError("All ENetUpscaled channels must be divisible by 4.")
        ch0, ch1, ch2, ch3, ch4 = (int(c) for c in channels)
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

        # Stage 1: stride-2 downsample to H/4, then 6 regular blocks [C2: +50%, 4→6]
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(6)])

        # [C3] Stage 2: flat at H/4 (NO downsampling), 12 blocks [C2: +50%, 8→12]
        self.proj1_to_2 = (
            nn.Identity()
            if ch2 == ch1
            else nn.Sequential(
                nn.Conv2d(ch1, ch2, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch2),
                nn.PReLU(ch2),
            )
        )
        self.stage2_flat = nn.Sequential(*_context_stage(ch2), *_extra_blocks(ch2))

        # [C3] Stage 3: downsampling now happens here (delayed from stage 2)
        # [C2] 12 blocks = 8 original context + 4 extra (+50%, 8→12)
        self.down2 = DownsamplingBottleneck(ch2, ch3, dropout_p=0.1)
        extra_fn = _asym_extra_blocks if asym_heavy_stage3 else _extra_blocks
        self.stage3 = nn.Sequential(*_context_stage(ch3), *extra_fn(ch3))

        # [C1/C6] New stage 4: additional downsample to H/16 + 4 context blocks
        self.down3 = DownsamplingBottleneck(ch3, ch4, dropout_p=0.1)
        self.stage4 = nn.Sequential(*_extra_blocks(ch4))

        # --- Decoder (MaxUnpool, paper-faithful) ---
        # out_channels of each UpsamplingBottleneck must equal in_channels of the
        # paired DownsamplingBottleneck so MaxUnpool indices have matching channel count.
        # down3(ch3→ch4) → up_deep out=ch3; down2(ch2→ch3) → up_mid out=ch2; down1(ch0→ch1) → up_shal out=ch0

        # [C6] H/16 → H/8
        self.up_deep = UpsamplingBottleneck(ch4, ch3)
        self.dec3 = nn.Sequential(
            RegularBottleneck(ch3, dropout_p=0.1, relu=True),
            RegularBottleneck(ch3, dropout_p=0.1, relu=True),
        )

        # H/8 → H/4
        self.up_mid = UpsamplingBottleneck(ch3, ch2)
        self.dec2 = nn.Sequential(
            RegularBottleneck(ch2, dropout_p=0.1, relu=True),
            RegularBottleneck(ch2, dropout_p=0.1, relu=True),
        )

        # H/4 → H/2
        self.up_shal = UpsamplingBottleneck(ch2, ch0)
        self.dec1 = RegularBottleneck(ch0, dropout_p=0.1, relu=True)

        # H/2 → H
        self.final = nn.ConvTranspose2d(ch0, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.initial(x)
        x = self.initial_extra(x)

        x, idx1, sz1 = self.down1(x)   # H/2 → H/4
        x = self.stage1(x)

        x = self.proj1_to_2(x)
        x = self.stage2_flat(x)         # flat at H/4

        x, idx2, sz2 = self.down2(x)   # H/4 → H/8
        x = self.stage3(x)

        x, idx3, sz3 = self.down3(x)   # H/8 → H/16
        x = self.stage4(x)

        # Decoder (MaxUnpool)
        x = self.up_deep(x, sz3, idx3)  # H/16 → H/8
        x = self.dec3(x)

        x = self.up_mid(x, sz2, idx2)   # H/8 → H/4
        x = self.dec2(x)

        x = self.up_shal(x, sz1, idx1)  # H/4 → H/2
        x = self.dec1(x)

        x = self.final(x)               # H/2 → H
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
