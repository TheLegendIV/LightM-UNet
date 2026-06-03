from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseConvNormAct(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SequenceMamba(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()

        b, c = x.shape[:2]
        assert c == self.channels
        spatial_shape = x.shape[2:]
        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(b, c, n_tokens).transpose(1, 2)
        x_flat = self.norm(x_flat)
        x_flat = self.mamba(x_flat)
        return x_flat.transpose(1, 2).reshape(b, c, *spatial_shape)


class PVMamba(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("PVMamba requires channels divisible by 4.")

        self.channels = channels
        self.chunk_channels = channels // 4
        self.norm_in = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)
        self.mamba_chunks = nn.ModuleList(
            [SequenceMamba(self.chunk_channels, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(4)]
        )
        self.alpha = nn.Parameter(torch.ones(4, 1, 1, 1))
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()

        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(b, c, n_tokens).transpose(1, 2)
        x_norm = self.norm_in(x_flat).transpose(1, 2).reshape(b, c, *spatial_shape)

        chunks = torch.chunk(x_norm, 4, dim=1)
        out_chunks = [
            mamba(chunk) + self.alpha[i] * chunk
            for i, (mamba, chunk) in enumerate(zip(self.mamba_chunks, chunks))
        ]
        out = torch.cat(out_chunks, dim=1)

        out_flat = out.reshape(b, c, n_tokens).transpose(1, 2)
        out_flat = self.proj(self.norm_out(out_flat))
        return out_flat.transpose(1, 2).reshape(b, c, *spatial_shape)


class EMA(nn.Module):
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.groups = max(1, min(groups, channels))
        while channels % self.groups != 0:
            self.groups -= 1
        group_channels = channels // self.groups

        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1, bias=False)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(1, group_channels)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        grouped = x.reshape(b * self.groups, c // self.groups, h, w)

        context_h = grouped.mean(dim=3, keepdim=True)
        context_w = grouped.mean(dim=2, keepdim=True).transpose(2, 3)
        context = torch.cat([context_h, context_w], dim=2)
        context = self.conv1x1(context)
        att_h, att_w = torch.split(context, [h, w], dim=2)
        att_w = att_w.transpose(2, 3)

        gated = self.norm(grouped * torch.sigmoid(att_h) * torch.sigmoid(att_w))
        local = self.conv3x3(grouped)
        att = torch.sigmoid((gated + local).mean(dim=(2, 3), keepdim=True))
        out = (grouped * att).reshape(b, c, h, w)
        return x + self.out(out)


class LMStage(nn.Module):
    def __init__(self, channels: int, use_pv_mamba: bool):
        super().__init__()
        self.local = DepthwiseConvNormAct(channels)
        self.global_block = PVMamba(channels) if use_pv_mamba else DepthwiseConvNormAct(channels)
        self.ema = EMA(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x)
        x = x + self.global_block(x)
        return self.ema(x)


class EdgeFeatureExtraction(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, edge_channels: int = 16):
        super().__init__()
        self.low_pv = PVMamba(low_channels)
        self.high_pv = PVMamba(high_channels)
        self.high_proj = nn.Conv2d(high_channels, low_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            ConvNormAct(low_channels * 2, edge_channels),
            ConvNormAct(edge_channels, edge_channels),
            PVMamba(edge_channels),
            nn.Conv2d(edge_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low = self.low_pv(low)
        high = self.high_pv(high)
        high = self.high_proj(high)
        high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([low, high], dim=1))


class EdgeFeatureFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = ConvNormAct(channels, channels)
        self.pv = PVMamba(channels)
        kernel_size = 3 if channels >= 3 else 1
        padding = kernel_size // 2
        self.channel_att = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = F.interpolate(edge, size=x.shape[2:], mode="bilinear", align_corners=False)
        fused = x + x * edge
        conv = self.conv(fused)
        pv = self.pv(conv)
        att = conv.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        att = torch.sigmoid(self.channel_att(att)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return pv * att


class MultiStageChannelAttention(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        total_channels = sum(channels)
        self.local = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.proj = nn.ModuleList([nn.Linear(total_channels, channel) for channel in channels])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        pooled = [feature.mean(dim=(2, 3)) for feature in features]
        merged = torch.cat(pooled, dim=1).unsqueeze(1)
        merged = self.local(merged).squeeze(1)
        out = []
        for feature, proj in zip(features, self.proj):
            att = torch.sigmoid(proj(merged)).unsqueeze(-1).unsqueeze(-1)
            out.append(feature + feature * att)
        return out


class MultiScaleSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=9, dilation=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_value = x.max(dim=1, keepdim=True)[0]
        att = torch.sigmoid(self.conv(torch.cat([avg, max_value], dim=1)))
        return x + x * att


class MMSC(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.eff = nn.ModuleList([EdgeFeatureFusion(channel) for channel in channels])
        self.mca = MultiStageChannelAttention(channels)
        self.mmsa = nn.ModuleList([MultiScaleSpatialAttention() for _ in channels])

    def forward(self, features: list[torch.Tensor], edge: torch.Tensor) -> list[torch.Tensor]:
        features = [module(feature, edge) for module, feature in zip(self.eff, features)]
        features = self.mca(features)
        return [module(feature) for module, feature in zip(self.mmsa, features)]


class LMDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_pv_mamba: bool):
        super().__init__()
        self.reduce = ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=1)
        self.stage = LMStage(out_channels, use_pv_mamba=use_pv_mamba)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.stage(self.reduce(x))


class LMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: tuple[int, ...] = (12, 20, 32, 44, 64, 72),
        edge_channels: int = 20,
    ):
        super().__init__()
        if len(channels) != 6:
            raise ValueError("LMUNet expects six encoder channel stages.")
        if any(channel % 4 != 0 for channel in channels):
            raise ValueError("All LMUNet channels must be divisible by 4 for PV-Mamba.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.edge_channels = edge_channels

        self.stem = ConvNormAct(in_channels, channels[0])
        self.downsamples = nn.ModuleList(
            [ConvNormAct(channels[i], channels[i + 1], stride=2) for i in range(len(channels) - 1)]
        )
        self.encoder_stages = nn.ModuleList(
            [LMStage(channel, use_pv_mamba=i >= 3) for i, channel in enumerate(channels)]
        )

        self.efe = EdgeFeatureExtraction(channels[1], channels[5], edge_channels=edge_channels)
        self.mmsc = MMSC(list(channels))

        decoder_specs = [
            (channels[5], channels[4], channels[4], True),
            (channels[4], channels[3], channels[3], True),
            (channels[3], channels[2], channels[2], False),
            (channels[2], channels[1], channels[1], False),
            (channels[1], channels[0], channels[0], False),
        ]
        self.decoder = nn.ModuleList([LMDecoderBlock(*spec) for spec in decoder_specs])
        self.final = nn.Conv2d(channels[0], out_channels, 1)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = self.encoder_stages[0](self.stem(x))
        features.append(x)
        for downsample, stage in zip(self.downsamples, self.encoder_stages[1:]):
            x = stage(downsample(x))
            features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        edge = self.efe(features[1], features[5])
        skips = self.mmsc(features, edge)

        x = skips[-1]
        for decoder_block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = decoder_block(x, skip)
        return self.final(x)
