from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.nets.ENetOriginal import (
    DownsamplingBottleneck,
    InitialBlock,
    RegularBottleneck,
    UpsamplingBottleneck,
)


class ENetModified(nn.Module):
    """Reduced ENet variant from the FPGA ENet paper.

    The modified paper removes asymmetric, dilated and strided internal convolutions,
    uses three bottlenecks per block, and controls model size with the six filter
    multiplicities f0..f5.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        filters: Sequence[int] = (32, 67, 69, 66, 67, 66),
    ):
        super().__init__()
        if len(filters) != 6:
            raise ValueError("ENetModified expects six filters: f0,f1,f2,f3,f4,f5.")
        if any(channel < 4 for channel in filters):
            raise ValueError("All ENetModified filter values must be at least 4.")

        f0, f1, f2, f3, f4, f5 = (int(channel) for channel in filters)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = (f0, f1, f2, f3, f4, f5)
        self.channels = self.filters

        self.initial = InitialBlock(in_channels, f0)
        self.block1_down = DownsamplingBottleneck(f0, f1, dropout_p=0.01)
        self.block1 = nn.Sequential(RegularBottleneck(f1, dropout_p=0.01),
                                    RegularBottleneck(f1, dropout_p=0.01))

        self.block2_down = DownsamplingBottleneck(f1, f2, dropout_p=0.1)
        self.block2 = nn.Sequential(RegularBottleneck(f2, dropout_p=0.1),
                                    RegularBottleneck(f2, dropout_p=0.1))

        self.block3 = nn.Sequential(RegularBottleneck(f3, dropout_p=0.1),
                                    RegularBottleneck(f3, dropout_p=0.1),
                                    RegularBottleneck(f3, dropout_p=0.1))
        self.proj2_to_3 = nn.Identity() if f2 == f3 else nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size=1, bias=False),
            nn.BatchNorm2d(f3),
            nn.PReLU(f3),
        )

        self.block4_up = UpsamplingBottleneck(f3, f4, relu=True)
        self.block4 = nn.Sequential(RegularBottleneck(f4, dropout_p=0.1, relu=True),
                                    RegularBottleneck(f4, dropout_p=0.1, relu=True))

        self.block5_up = UpsamplingBottleneck(f4, f5, relu=True)
        self.block5 = nn.Sequential(RegularBottleneck(f5, dropout_p=0.1, relu=True),
                                    RegularBottleneck(f5, dropout_p=0.1, relu=True))
        self.final = nn.ConvTranspose2d(f5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.initial(x)
        x, _, size1 = self.block1_down(x)
        x = self.block1(x)
        x, _, size2 = self.block2_down(x)
        x = self.block2(x)
        x = self.proj2_to_3(x)
        x = self.block3(x)
        x = self.block4_up(x, size2)
        x = self.block4(x)
        x = self.block5_up(x, size1)
        x = self.block5(x)
        x = self.final(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
