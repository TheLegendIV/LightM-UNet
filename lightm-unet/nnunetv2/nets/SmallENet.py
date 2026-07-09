from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

"""
Small, full-resolution ENet variant for the ARCADE grid patches
(dataset-prep/prepare_arcade.py, e.g. Dataset501_ARCADE_6x6_1c).

    input (1ch grayscale)                                   [+margin, any HxW]
      -> concat local-contrast-norm channel                  2ch, full res
      -> initial block, stride 1 (no downsample)             16ch, full res
      -> bottleneck x2 (regular, 3x3)                        32ch, full res
      -> bottleneck (dilated 2)                               32ch
      -> bottleneck (dilated 4)                               32ch
      -> bottleneck (dilated 8)                               32ch   <- receptive field grows, no pooling
      -> bottleneck (asymmetric 5)                            32ch
      -> 1x1 conv head                                        1ch logits

There is no pooling/downsampling anywhere, so the network is fully
resolution-agnostic: patches can be fed with extra context margin and the
output cropped back to the core region afterward with crop_margin().

The head returns raw logits, not sigmoid probabilities -- apply
torch.sigmoid(...) externally for inference. nnU-Net's DC_and_BCE_loss
requires unnormalized logits (see its docstring), and applying sigmoid here
would double up with that loss's internal sigmoid.
"""


class LocalContrastNorm(nn.Module):
    """Parameter-free local contrast normalization: (x - local_mean) / local_std,
    computed with a reflect-padded box filter so output size matches input size."""

    def __init__(self, kernel_size: int = 9, eps: float = 1e-5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("LocalContrastNorm kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, [self.padding] * 4, mode="reflect")
        local_mean = F.avg_pool2d(padded, self.kernel_size, stride=1)
        local_sqmean = F.avg_pool2d(padded * padded, self.kernel_size, stride=1)
        local_var = (local_sqmean - local_mean * local_mean).clamp_min(0.0)
        local_std = torch.sqrt(local_var + self.eps)
        return (x - local_mean) / local_std


class SmallENetBottleneck(nn.Module):
    """ENet-style bottleneck (1x1 reduce -> [regular|dilated|asymmetric] conv ->
    1x1 expand, residual add) generalized with a learned 1x1 projection shortcut
    so it can also change channel count without any pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        internal_channels = max(out_channels // internal_ratio, 1)
        padding = dilation * (kernel_size // 2)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )

        if asymmetric:
            conv = nn.Sequential(
                nn.Conv2d(
                    internal_channels, internal_channels,
                    kernel_size=(kernel_size, 1), padding=(padding, 0), bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
                nn.Conv2d(
                    internal_channels, internal_channels,
                    kernel_size=(1, kernel_size), padding=(0, padding), bias=False,
                ),
            )
        else:
            conv = nn.Conv2d(
                internal_channels, internal_channels,
                kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False,
            )

        self.conv_bn_act = nn.Sequential(conv, nn.BatchNorm2d(internal_channels), nn.PReLU(internal_channels))
        self.expand = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.out_act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        return self.out_act(self.shortcut(x) + out)


class SmallENet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        initial_channels: int = 16,
        stage_channels: int = 32,
        lcn_kernel_size: int = 9,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "SmallENet derives its second input channel via local-contrast "
                f"normalization of a single grayscale channel; in_channels must be 1, got {in_channels}."
            )

        self.lcn = LocalContrastNorm(kernel_size=lcn_kernel_size)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + 1, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.PReLU(initial_channels),
        )
        self.bottleneck1 = SmallENetBottleneck(initial_channels, stage_channels, dropout_p=dropout_p)
        self.bottleneck2 = SmallENetBottleneck(stage_channels, stage_channels, dropout_p=dropout_p)
        self.dilated2 = SmallENetBottleneck(stage_channels, stage_channels, dilation=2, dropout_p=dropout_p)
        self.dilated4 = SmallENetBottleneck(stage_channels, stage_channels, dilation=4, dropout_p=dropout_p)
        self.dilated8 = SmallENetBottleneck(stage_channels, stage_channels, dilation=8, dropout_p=dropout_p)
        self.asymmetric5 = SmallENetBottleneck(
            stage_channels, stage_channels, kernel_size=5, asymmetric=True, dropout_p=dropout_p
        )
        self.head = nn.Conv2d(stage_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, self.lcn(x)], dim=1)
        x = self.initial(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.dilated2(x)
        x = self.dilated4(x)
        x = self.dilated8(x)
        x = self.asymmetric5(x)
        return self.head(x)


def crop_margin(x: torch.Tensor, margin: int) -> torch.Tensor:
    """Crop a fixed margin off all spatial sides, e.g. to recover the core
    patch region after running inference on a margin-padded input."""
    if margin <= 0:
        return x
    return x[..., margin:-margin, margin:-margin]
