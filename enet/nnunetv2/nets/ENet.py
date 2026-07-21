from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("InitialBlock out_channels must exceed in_channels.")
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class RegularBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout_p: float = 0.1,
        relu: bool = False,
    ):
        super().__init__()
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            activation(internal_channels) if not relu else activation(inplace=True),
        )

        if asymmetric:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(padding, 0),
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(internal_channels) if not relu else activation(inplace=True),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, padding),
                    bias=False,
                ),
            )
        else:
            self.conv = nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )

        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_channels),
            activation(internal_channels) if not relu else activation(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.out_act = activation(channels) if not relu else activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        return self.out_act(x + out)


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4, dropout_p: float = 0.01):
        super().__init__()
        internal_channels = out_channels // internal_ratio
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.out_act = nn.PReLU(out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        input_size = x.size()
        main, indices = self.pool(x)
        if main.shape[1] < self.out_channels:
            padding = torch.zeros(
                main.shape[0],
                self.out_channels - main.shape[1],
                main.shape[2],
                main.shape[3],
                dtype=main.dtype,
                device=main.device,
            )
            main = torch.cat([main, padding], dim=1)

        out = self.reduce(x)
        out = self.conv(out)
        out = self.dropout(self.expand(out))
        return self.out_act(main + out), indices, input_size


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4, relu: bool = True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.main_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True) if relu else nn.PReLU(internal_channels),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True) if relu else nn.PReLU(internal_channels),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.out_act = nn.ReLU(inplace=True) if relu else nn.PReLU(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        output_size: torch.Size,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        main = self.main_proj(x)
        if indices is None:
            main = F.interpolate(main, size=output_size[2:], mode="bilinear", align_corners=False)
        else:
            if main.shape[1] != indices.shape[1]:
                raise RuntimeError(
                    "ENet max-unpool channel mismatch: "
                    f"projected main branch has {main.shape[1]} channels but pooling indices have "
                    f"{indices.shape[1]}. This should not happen for the paper-faithful ENet layout."
                )
            try:
                main = self.unpool(main, indices, output_size=output_size)
            except RuntimeError as error:
                raise RuntimeError(
                    "ENet max-unpool failed. The paper-faithful decoder requires the nnU-Net "
                    "patch size to align with ENet's pooling/unpooling path. For this architecture, use "
                    "2D patch sizes whose spatial dimensions stay compatible after the initial stride-2 "
                    "block and two bottleneck downsamplings, such as 512x512. "
                    f"Input to unpool: {tuple(x.shape)}, projected branch: {tuple(main.shape)}, "
                    f"indices: {tuple(indices.shape)}, requested output_size: {tuple(output_size)}."
                ) from error
        out = self.reduce(x)
        out = self.up(out)
        out = self.dropout(self.expand(out))
        if out.shape[2:] != main.shape[2:]:
            raise RuntimeError(
                "ENet upsampling branch shape mismatch. The paper-faithful decoder produced "
                f"conv-transpose shape {tuple(out.shape)} but max-unpool main branch shape "
                f"{tuple(main.shape)}. This usually means the nnU-Net patch size is not compatible "
                "with ENet's fixed downsampling/upsampling stages."
            )
        return self.out_act(main + out)


class ENet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, int, int, int, int] = (20, 72, 144, 72, 20),
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("ENet expects five channel values: initial, stage1, stage2/3, stage4, stage5.")
        initial_channels, stage1_channels, stage23_channels, stage4_channels, stage5_channels = channels
        if stage1_channels % 4 != 0 or stage23_channels % 4 != 0 or stage4_channels % 4 != 0 or stage5_channels % 4 != 0:
            raise ValueError("ENet stage channels must be divisible by 4 for bottleneck reduction.")
        if initial_channels <= in_channels:
            raise ValueError("ENet initial channels must exceed input channels.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(int(channel) for channel in channels)

        self.initial = self._build_initial_block(in_channels, initial_channels)
        self.down1 = DownsamplingBottleneck(initial_channels, stage1_channels, dropout_p=0.01)
        self.regular1 = nn.Sequential(*[RegularBottleneck(stage1_channels, dropout_p=0.01) for _ in range(4)])

        self.down2 = DownsamplingBottleneck(stage1_channels, stage23_channels, dropout_p=0.1)
        def make_context_stage() -> nn.Sequential:
            return nn.Sequential(
                RegularBottleneck(stage23_channels, dropout_p=0.1),
                RegularBottleneck(stage23_channels, padding=2, dilation=2, dropout_p=0.1),
                RegularBottleneck(stage23_channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
                RegularBottleneck(stage23_channels, padding=4, dilation=4, dropout_p=0.1),
                RegularBottleneck(stage23_channels, dropout_p=0.1),
                RegularBottleneck(stage23_channels, padding=8, dilation=8, dropout_p=0.1),
                RegularBottleneck(stage23_channels, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
                RegularBottleneck(stage23_channels, padding=16, dilation=16, dropout_p=0.1),
            )

        self.stage2 = make_context_stage()
        self.stage3 = make_context_stage()
        self.up4 = UpsamplingBottleneck(stage23_channels, stage4_channels)
        self.regular4 = nn.Sequential(RegularBottleneck(stage4_channels, dropout_p=0.1, relu=True),
                                      RegularBottleneck(stage4_channels, dropout_p=0.1, relu=True))
        self.up5 = UpsamplingBottleneck(stage4_channels, stage5_channels)
        self.regular5 = RegularBottleneck(stage5_channels, dropout_p=0.1, relu=True)
        self.final = nn.ConvTranspose2d(stage5_channels, out_channels, kernel_size=2, stride=2)

    def _build_initial_block(self, in_channels: int, initial_channels: int) -> nn.Module:
        """Overridable hook so subclasses can swap in a different first stage
        (e.g. ENetPostRefinement.py's two-stem variant) without duplicating
        the rest of __init__ -- everything downstream of self.initial is
        architecture-agnostic to how it got built, it just needs to receive
        (in_channels, H/2, W/2) and emit (initial_channels, H/2, W/2)."""
        return InitialBlock(in_channels, initial_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.initial(x)
        x, indices1, size1 = self.down1(x)
        x = self.regular1(x)
        x, indices2, size2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, size2, indices2)
        x = self.regular4(x)
        x = self.up5(x, size1, indices1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
