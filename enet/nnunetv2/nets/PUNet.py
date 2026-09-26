from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm import Mamba


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DSConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ChunkChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = x.mean(dim=(2, 3))
        max_value = x.amax(dim=(2, 3))
        att = torch.sigmoid(self.mlp(avg) + self.mlp(max_value)).view(b, c, 1, 1)
        return x * att


class VMLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        reduction: int = 8,
    ):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("VMLayer requires channels divisible by 4.")
        self.channels = channels
        self.chunk_channels = channels // 4
        self.norm = nn.LayerNorm(channels)
        self.mambas = nn.ModuleList(
            [
                Mamba(
                    d_model=self.chunk_channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(4)
            ]
        )
        self.skip_scale = nn.Parameter(torch.ones(4, 1, 1))
        self.attention = nn.ModuleList(
            [ChunkChannelAttention(self.chunk_channels, reduction=reduction) for _ in range(4)]
        )
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        b, c = x.shape[:2]
        if c != self.channels:
            raise ValueError(f"VMLayer expected {self.channels} channels, got {c}.")
        spatial_shape = x.shape[2:]
        n_tokens = x.shape[2:].numel()

        seq = x.reshape(b, c, n_tokens).transpose(1, 2)
        seq = self.norm(seq)
        chunks = torch.chunk(seq, 4, dim=2)

        outputs = []
        for idx, (chunk, mamba, attention) in enumerate(zip(chunks, self.mambas, self.attention)):
            out = mamba(chunk) + self.skip_scale[idx] * chunk
            out = out.transpose(1, 2).reshape(b, self.chunk_channels, *spatial_shape)
            outputs.append(attention(out))

        fused = torch.cat(outputs, dim=1)
        fused_seq = fused.reshape(b, c, n_tokens).transpose(1, 2)
        fused_seq = self.proj(fused_seq)
        return fused_seq.transpose(1, 2).reshape(b, c, *spatial_shape)


class PSSSE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 8):
        super().__init__()
        self.local = DSConv(in_channels, out_channels, stride=stride)
        self.global_proj = DSConv(in_channels, out_channels, stride=stride)
        self.global_block = VMLayer(out_channels, reduction=reduction)
        self.local_se = SEModule(out_channels, reduction=reduction)
        self.global_se = SEModule(out_channels, reduction=reduction)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_se(self.local(x))
        global_feature = self.global_se(self.global_block(self.global_proj(x)))
        return self.fuse(torch.cat([local, global_feature], dim=1))


class SESkipFusion(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = SEModule(channels, reduction=reduction)
        self.refine = DSConv(channels, channels)

    def forward(self, decoder_feature: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoder_feature = F.interpolate(decoder_feature, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.se(skip)
        return self.refine(decoder_feature + skip)


class PUDecoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.vm = VMLayer(out_channels, reduction=reduction)
        self.skip = SESkipFusion(out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.vm(x)
        return self.skip(x, skip)


class PUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: Sequence[int] = (16, 24, 36, 52, 72, 104),
        reduction: int = 8,
    ):
        super().__init__()
        if len(channels) != 6:
            raise ValueError("PUNet expects six channel stages.")
        if any(channel % 4 != 0 for channel in channels):
            raise ValueError("All PUNet channels must be divisible by 4 for VMLayer chunking.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(int(channel) for channel in channels)

        c = self.channels
        self.encoder = nn.ModuleList()
        prev = in_channels
        for idx, channel in enumerate(c):
            self.encoder.append(PSSSE(prev, channel, stride=1 if idx == 0 else 2, reduction=reduction))
            prev = channel

        self.decoder = nn.ModuleList(
            [
                PUDecoderStage(c[5], c[4], reduction=reduction),
                PUDecoderStage(c[4], c[3], reduction=reduction),
                PUDecoderStage(c[3], c[2], reduction=reduction),
                PUDecoderStage(c[2], c[1], reduction=reduction),
                PUDecoderStage(c[1], c[0], reduction=reduction),
            ]
        )
        self.final_refine = PSSSE(c[0], c[0], stride=1, reduction=reduction)
        self.final = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        skips = []
        out = x
        for stage in self.encoder:
            out = stage(out)
            skips.append(out)

        out = skips[-1]
        for stage, skip in zip(self.decoder, reversed(skips[:-1])):
            out = stage(out, skip)
        out = self.final_refine(out)
        out = self.final(out)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out
