from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet101_Weights, resnet101
except ImportError as e:
    raise ImportError(
        "TVResNet101UNet requires torchvision. Install torchvision in the environment "
        "used for nnU-Net training."
    ) from e


def _group_norm(num_channels: int) -> nn.GroupNorm:
    for num_groups in (32, 16, 8, 4, 2):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class TVResNet101UNet(nn.Module):
    """
    2D U-Net-style segmentation model with a torchvision ResNet101 encoder.

    The encoder exposes feature maps at strides 2, 4, 8, 16 and 32. The decoder
    upsamples back to the original input size and returns raw segmentation logits.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_weights: Optional[str] = "IMAGENET1K_V2",
        decoder_channels: tuple[int, int, int, int, int] = (512, 256, 128, 64, 32),
    ):
        super().__init__()

        if len(decoder_channels) != 5:
            raise ValueError("decoder_channels must contain five channel sizes.")

        weights = None
        if encoder_weights is not None:
            weights = ResNet101_Weights[encoder_weights]

        encoder = resnet101(weights=weights)
        self._adapt_first_conv(encoder, in_channels)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4 = UpBlock(2048, 1024, decoder_channels[0])
        self.up3 = UpBlock(decoder_channels[0], 512, decoder_channels[1])
        self.up2 = UpBlock(decoder_channels[1], 256, decoder_channels[2])
        self.up1 = UpBlock(decoder_channels[2], 64, decoder_channels[3])
        self.up0 = ConvBlock(decoder_channels[3], decoder_channels[4])
        self.seg_head = nn.Conv2d(decoder_channels[4], out_channels, kernel_size=1)

    @staticmethod
    def _adapt_first_conv(encoder: nn.Module, in_channels: int) -> None:
        if in_channels == encoder.conv1.in_channels:
            return

        old_conv = encoder.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            elif in_channels > 3:
                repeat_count = (in_channels + 2) // 3
                weight = old_conv.weight.repeat(1, repeat_count, 1, 1)[:, :in_channels]
                weight = weight * (3.0 / float(in_channels))
                new_conv.weight.copy_(weight)
            else:
                new_conv.weight.copy_(old_conv.weight[:, :in_channels])
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        encoder.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"TVResNet101UNet expects 2D input [B, C, H, W], got shape {tuple(x.shape)}.")

        input_size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.up0(x)
        return self.seg_head(x)
