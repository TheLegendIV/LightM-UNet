from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class PVMLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError("PVMLayer requires input_dim divisible by 4.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        b, c = x.shape[:2]
        if c != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input channels, got {c}.")
        spatial_shape = x.shape[2:]
        n_tokens = x.shape[2:].numel()
        x_flat = x.reshape(b, c, n_tokens).transpose(1, 2)
        x_norm = self.norm(x_flat)

        chunks = torch.chunk(x_norm, 4, dim=2)
        chunks = [self.mamba(chunk) + self.skip_scale * chunk for chunk in chunks]
        x_mamba = torch.cat(chunks, dim=2)

        x_mamba = self.proj(self.norm(x_mamba))
        return x_mamba.transpose(1, 2).reshape(b, self.output_dim, *spatial_shape)


class ChannelAttentionBridge(nn.Module):
    def __init__(self, channels: tuple[int, ...], split_att: str = "fc"):
        super().__init__()
        channels_sum = sum(channels) - channels[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att = nn.ModuleList(
            [
                nn.Linear(channels_sum, channel) if split_att == "fc" else nn.Conv1d(channels_sum, channel, 1)
                for channel in channels[:-1]
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, *features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        att = torch.cat([self.avgpool(feature) for feature in features], dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(1, 2))
        if self.split_att != "fc":
            att = att.transpose(1, 2)

        out = []
        for feature, layer in zip(features, self.att):
            feature_att = self.sigmoid(layer(att))
            if self.split_att == "fc":
                feature_att = feature_att.transpose(1, 2).unsqueeze(-1).expand_as(feature)
            else:
                feature_att = feature_att.unsqueeze(-1).expand_as(feature)
            out.append(feature_att)
        return tuple(out)


class SpatialAttentionBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=9, dilation=3),
            nn.Sigmoid(),
        )

    def forward(self, *features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = []
        for feature in features:
            avg_out = torch.mean(feature, dim=1, keepdim=True)
            max_out = torch.max(feature, dim=1, keepdim=True)[0]
            out.append(self.shared_conv2d(torch.cat([avg_out, max_out], dim=1)))
        return tuple(out)


class SCAttentionBridge(nn.Module):
    def __init__(self, channels: tuple[int, ...], split_att: str = "fc"):
        super().__init__()
        self.catt = ChannelAttentionBridge(channels, split_att=split_att)
        self.satt = SpatialAttentionBridge()

    def forward(self, *features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        residuals = features
        spatial_att = self.satt(*features)
        spatial_features = tuple(att * feature for att, feature in zip(spatial_att, features))
        features = tuple(feature + residual for feature, residual in zip(spatial_features, residuals))
        channel_att = self.catt(*features)
        features = tuple(att * feature for att, feature in zip(channel_att, features))
        return tuple(feature + residual for feature, residual in zip(features, spatial_features))


class UltraLightVMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: tuple[int, ...] = (8, 16, 24, 32, 48, 64),
        split_att: str = "fc",
        bridge: bool = True,
    ):
        super().__init__()
        if len(channels) != 6:
            raise ValueError("UltraLightVMUNet expects six channel stages.")
        if any(channel % 4 != 0 for channel in channels):
            raise ValueError("All UltraLightVMUNet channels must be divisible by 4.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.bridge = bridge

        self.encoder1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, padding=1)
        self.encoder4 = PVMLayer(input_dim=channels[2], output_dim=channels[3])
        self.encoder5 = PVMLayer(input_dim=channels[3], output_dim=channels[4])
        self.encoder6 = PVMLayer(input_dim=channels[4], output_dim=channels[5])

        if bridge:
            self.scab = SCAttentionBridge(channels, split_att)

        self.decoder1 = PVMLayer(input_dim=channels[5], output_dim=channels[4])
        self.decoder2 = PVMLayer(input_dim=channels[4], output_dim=channels[3])
        self.decoder3 = PVMLayer(input_dim=channels[3], output_dim=channels[2])
        self.decoder4 = nn.Conv2d(channels[2], channels[1], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(channels[1], channels[0], 3, stride=1, padding=1)

        self.ebn1 = nn.GroupNorm(4, channels[0])
        self.ebn2 = nn.GroupNorm(4, channels[1])
        self.ebn3 = nn.GroupNorm(4, channels[2])
        self.ebn4 = nn.GroupNorm(4, channels[3])
        self.ebn5 = nn.GroupNorm(4, channels[4])
        self.dbn1 = nn.GroupNorm(4, channels[4])
        self.dbn2 = nn.GroupNorm(4, channels[3])
        self.dbn3 = nn.GroupNorm(4, channels[2])
        self.dbn4 = nn.GroupNorm(4, channels[1])
        self.dbn5 = nn.GroupNorm(4, channels[0])

        self.final = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            n = module.kernel_size[0] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2.0 / n))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out))) + t5
        out4 = F.interpolate(self.dbn2(self.decoder2(out5)), size=t4.shape[2:], mode="bilinear", align_corners=False)
        out4 = F.gelu(out4) + t4
        out3 = F.interpolate(self.dbn3(self.decoder3(out4)), size=t3.shape[2:], mode="bilinear", align_corners=False)
        out3 = F.gelu(out3) + t3
        out2 = F.interpolate(self.dbn4(self.decoder4(out3)), size=t2.shape[2:], mode="bilinear", align_corners=False)
        out2 = F.gelu(out2) + t2
        out1 = F.interpolate(self.dbn5(self.decoder5(out2)), size=t1.shape[2:], mode="bilinear", align_corners=False)
        out1 = F.gelu(out1) + t1

        out0 = self.final(out1)
        return F.interpolate(out0, size=input_size, mode="bilinear", align_corners=False)
