from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

DecoderType = Literal["max_unpool", "upsample_conv"]
ContextPattern = Literal["default", "sparse"]

# The ENet-native context-stage pattern (Paszke et al.): regular, dilated x2,
# asymmetric 5x5, dilated x4, regular, dilated x8, asymmetric 5x5, dilated x16.
# bottlenecks_per_stage truncates this to its first n entries; use_dilated /
# use_asymmetric downgrade the matching slots to a plain regular bottleneck
# instead of removing them, so stage depth (n) and op composition are
# independent knobs.
CONTEXT_STAGE_PATTERN: tuple[dict, ...] = (
    {},
    {"padding": 2, "dilation": 2},
    {"kernel_size": 5, "padding": 2, "asymmetric": True},
    {"padding": 4, "dilation": 4},
    {},
    {"padding": 8, "dilation": 8},
    {"kernel_size": 5, "padding": 2, "asymmetric": True},
    {"padding": 16, "dilation": 16},
)

# Sparse dilation-only context pattern (section 2a's reduced-depth
# bottleneck axis, div2/div4): regular, dilated x4, regular, dilated x16 --
# skips the 2/8 rungs and never uses asymmetric convs at all (the grid runs
# with use_asymmetric=0 anyway, but this pattern doesn't rely on that flag
# to get there -- it simply never emits an asymmetric slot). 4 entries,
# repeats via i % len(pattern) if n_ops > 4 (not exercised by 2a's grid,
# which caps at 4, but kept general like CONTEXT_STAGE_PATTERN).
SPARSE_DILATION_PATTERN: tuple[dict, ...] = (
    {},
    {"padding": 4, "dilation": 4},
    {},
    {"padding": 16, "dilation": 16},
)


def _activation(channels: int, relu: bool) -> nn.Module:
    return nn.ReLU(inplace=True) if relu else nn.PReLU(channels)


class InitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu: bool = False):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("InitialBlock out_channels must exceed in_channels.")
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _activation(out_channels, relu)

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
        use_dsc: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)
        activation = nn.ReLU if relu else nn.PReLU

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            activation(internal_channels) if not relu else activation(inplace=True),
        )

        if asymmetric:
            if use_dsc:
                raise ValueError(
                    "use_dsc is not defined for asymmetric bottlenecks -- asymmetric is "
                    "itself a factorization of the inner conv, not exercised in combination "
                    "with DSC by any planned experiment. Disable use_asymmetric to use DSC."
                )
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
        elif use_dsc:
            # Depthwise separable: depthwise k x k (groups=internal_channels,
            # one filter per channel, no cross-channel mixing) + pointwise
            # 1x1 (mixes channels back). Standard MobileNet-style
            # factorization of the plain/dilated regular conv -- dilation
            # still applies to the depthwise stage, pointwise is unaffected
            # by it.
            self.conv = nn.Sequential(
                nn.Conv2d(
                    internal_channels, internal_channels, kernel_size=kernel_size,
                    padding=padding, dilation=dilation, groups=internal_channels, bias=False,
                ),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=1, bias=False),
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        dropout_p: float = 0.01,
        use_strided: bool = True,
        relu: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, out_channels // internal_ratio)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        if use_strided:
            # Spatial downsampling folded into the reduce conv itself (single
            # stride-2 2x2 conv) -- the ENet-native / FINN-favoured path.
            self.reduce = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(internal_channels),
                _activation(internal_channels, relu),
            )
        else:
            # Stage-1b ablation: separate maxpool (no learned downsampling)
            # followed by a stride-1 1x1 conv, instead of a strided conv.
            self.reduce = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(internal_channels),
                _activation(internal_channels, relu),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.out_act = _activation(out_channels, relu)
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
        elif main.shape[1] > self.out_channels:
            # Several compression-sweep filter configs (U8/U16/UF) have
            # stage1_channels < initial_channels, i.e. this block's main
            # branch must shrink rather than grow -- the paper's ENet never
            # does this (channels only ever expand downsampling in from 16),
            # so this case didn't exist before the sweep. Truncating keeps
            # the main branch parameter-free, matching the zero-pad case's
            # "no learned params on the identity path" design.
            main = main[:, : self.out_channels]

        out = self.reduce(x)
        out = self.conv(out)
        out = self.dropout(self.expand(out))
        return self.out_act(main + out), indices, input_size


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4, relu: bool = True):
        super().__init__()
        internal_channels = max(1, in_channels // internal_ratio)
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
                    "ENet max-unpool failed. The max_unpool decoder requires the nnU-Net "
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
                "ENet upsampling branch shape mismatch. The decoder produced "
                f"conv-transpose shape {tuple(out.shape)} but main branch shape "
                f"{tuple(main.shape)}. This usually means the nnU-Net patch size is not compatible "
                "with ENet's fixed downsampling/upsampling stages."
            )
        return self.out_act(main + out)


class ENet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, ...] = (20, 72, 144, 72, 20),
        bottlenecks_per_stage: tuple[int, int, int, int, int] = (4, 8, 8, 2, 1),
        decoder_type: DecoderType = "max_unpool",
        use_dilated: bool = True,
        use_asymmetric: bool = True,
        use_strided: bool = True,
        use_dsc: bool = False,
        context_pattern: ContextPattern = "default",
        use_prelu: bool = True,
    ):
        super().__init__()
        if context_pattern not in ("default", "sparse"):
            raise ValueError(f"context_pattern must be 'default' or 'sparse', got {context_pattern!r}.")
        if len(channels) == 5:
            # stage2 and stage3 share one width (the historical/default
            # convention throughout this project's compression sweep).
            initial_channels, stage1_channels, stage2_channels, stage4_channels, stage5_channels = channels
            stage3_channels = stage2_channels
        elif len(channels) == 6:
            # Split stage2/stage3 widths -- (initial, stage1, stage2, stage3,
            # stage4, stage5). Only used by upscale/'s combinatorial sweep so
            # far; unrelated to max_unpool's f_i==f5/f1==f4 constraint (that
            # constraint comes from down1/down2's pooling INPUT widths --
            # initial_channels and stage1_channels -- matching up5/up4's
            # projected OUTPUT widths, never stage2/stage3, which sit
            # entirely between down2 and up4 with no pooling of their own).
            initial_channels, stage1_channels, stage2_channels, stage3_channels, stage4_channels, stage5_channels = channels
        else:
            raise ValueError(
                "ENet expects five channel values (initial, stage1, stage2/3, stage4, stage5), or six if "
                "stage2/stage3 widths are split (initial, stage1, stage2, stage3, stage4, stage5)."
            )
        if len(bottlenecks_per_stage) != 5:
            raise ValueError(
                "ENet expects five bottleneck-count values: stage1, stage2, stage3, stage4(regular4), "
                "stage5(regular5)."
            )
        if (stage1_channels % 4 != 0 or stage2_channels % 4 != 0 or stage3_channels % 4 != 0
                or stage4_channels % 4 != 0 or stage5_channels % 4 != 0):
            raise ValueError("ENet stage channels must be divisible by 4 for bottleneck reduction.")
        if initial_channels <= in_channels:
            raise ValueError("ENet initial channels must exceed input channels.")
        if decoder_type not in ("max_unpool", "upsample_conv"):
            raise ValueError(f"decoder_type must be 'max_unpool' or 'upsample_conv', got {decoder_type!r}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(int(channel) for channel in channels)
        self.bottlenecks_per_stage = tuple(int(n) for n in bottlenecks_per_stage)
        self.decoder_type: DecoderType = decoder_type
        self.use_dilated = use_dilated
        self.use_asymmetric = use_asymmetric
        self.use_strided = use_strided
        self.use_dsc = use_dsc
        self.context_pattern: ContextPattern = context_pattern
        # Encoder activation: PReLU by default (paper-faithful, see
        # RegularBottleneck/UpsamplingBottleneck's own `relu` flags, which
        # already hardcode the decoder half -- regular4/regular5/up4/up5 --
        # to plain ReLU regardless of this flag, matching the ENet paper's
        # empirical PReLU-encoder/ReLU-decoder split). use_prelu=False
        # switches the encoder to ReLU too, collapsing the whole network to
        # a single activation, for section 1d's ablation of that split.
        self.use_prelu = use_prelu

        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = self.bottlenecks_per_stage

        self.initial = self._build_initial_block(in_channels, initial_channels)
        self.down1 = DownsamplingBottleneck(
            initial_channels, stage1_channels, dropout_p=0.01, use_strided=use_strided,
            relu=not use_prelu,
        )
        self.regular1 = nn.Sequential(
            *[RegularBottleneck(stage1_channels, dropout_p=0.01, use_dsc=use_dsc, relu=not use_prelu)
              for _ in range(n_stage1)]
        )

        self.down2 = DownsamplingBottleneck(
            stage1_channels, stage2_channels, dropout_p=0.1, use_strided=use_strided,
            relu=not use_prelu,
        )
        self.stage2 = self._make_context_stage(stage2_channels, n_stage2)
        self.proj2_to_3 = (
            nn.Identity()
            if stage2_channels == stage3_channels
            else nn.Sequential(
                nn.Conv2d(stage2_channels, stage3_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(stage3_channels),
                _activation(stage3_channels, not use_prelu),
            )
        )
        self.stage3 = self._make_context_stage(stage3_channels, n_stage3)

        self.up4 = UpsamplingBottleneck(stage3_channels, stage4_channels)
        self.regular4 = nn.Sequential(
            *[RegularBottleneck(stage4_channels, dropout_p=0.1, relu=True, use_dsc=use_dsc) for _ in range(n_regular4)]
        )
        self.up5 = UpsamplingBottleneck(stage4_channels, stage5_channels)
        self.regular5 = nn.Sequential(
            *[RegularBottleneck(stage5_channels, dropout_p=0.1, relu=True, use_dsc=use_dsc) for _ in range(n_regular5)]
        )
        self.final = nn.ConvTranspose2d(stage5_channels, out_channels, kernel_size=2, stride=2)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Migrates checkpoints trained before regular5 became parametric
        (bottlenecks_per_stage) -- it used to be a single bare
        RegularBottleneck ("regular5.reduce...."), not an
        nn.Sequential ("regular5.0.reduce...."). Stage 1's baselines
        (E1/enet_paper) predate that change. Pure rename, not an
        architecture change (identical computation when regular5 has
        exactly 1 rep, which both do) -- migrate rather than refuse to
        load real completed training runs. regular1/regular4 were already
        Sequential-wrapped before this session's changes, so they need no
        migration.
        """
        migrated = {}
        for key, value in state_dict.items():
            parts = key.split(".")
            if len(parts) > 1 and parts[0] == "regular5" and not parts[1].isdigit():
                key = "regular5.0." + ".".join(parts[1:])
            migrated[key] = value
        return super().load_state_dict(migrated, strict=strict)

    def _make_context_stage(self, channels: int, n_ops: int) -> nn.Sequential:
        """Builds an n_ops-long context stage from the first n_ops entries of
        CONTEXT_STAGE_PATTERN (or SPARSE_DILATION_PATTERN when
        context_pattern="sparse" -- section 2a's reduced-depth bottleneck
        axis, div2/div4: regular/dilated4/regular/dilated16, no 2/8 rungs,
        never asymmetric). use_dilated/use_asymmetric downgrade the matching
        slots to a plain regular bottleneck rather than dropping them, so
        stage depth and op composition stay independent knobs (Stage 2
        sweeps depth; Stage 1b's op ablation sweeps composition). use_dsc
        factorizes whatever inner conv results (plain or dilated -- not
        asymmetric, RegularBottleneck rejects that combination)."""
        pattern = SPARSE_DILATION_PATTERN if self.context_pattern == "sparse" else CONTEXT_STAGE_PATTERN
        ops = []
        for i in range(n_ops):
            kwargs = dict(pattern[i % len(pattern)])
            if kwargs.get("dilation", 1) != 1 and not self.use_dilated:
                kwargs = {}
            if kwargs.get("asymmetric", False) and not self.use_asymmetric:
                kwargs = {}
            ops.append(RegularBottleneck(channels, dropout_p=0.1, use_dsc=self.use_dsc, relu=not self.use_prelu, **kwargs))
        return nn.Sequential(*ops)

    def _build_initial_block(self, in_channels: int, initial_channels: int) -> nn.Module:
        """Overridable hook so subclasses can swap in a different first stage
        (e.g. ENetPostRefinement.py's two-stem variant) without duplicating
        the rest of __init__ -- everything downstream of self.initial is
        architecture-agnostic to how it got built, it just needs to receive
        (in_channels, H/2, W/2) and emit (initial_channels, H/2, W/2)."""
        return InitialBlock(in_channels, initial_channels, relu=not self.use_prelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.initial(x)
        x, indices1, size1 = self.down1(x)
        x = self.regular1(x)
        x, indices2, size2 = self.down2(x)
        x = self.stage2(x)
        x = self.proj2_to_3(x)
        x = self.stage3(x)
        use_indices = self.decoder_type == "max_unpool"
        x = self.up4(x, size2, indices2 if use_indices else None)
        x = self.regular4(x)
        x = self.up5(x, size1, indices1 if use_indices else None)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    # Foundation self-test: every filter config x bottleneck-depth x decoder
    # type x op-flag combination that the compression sweep plans to build
    # must at least construct and do one forward pass without shape errors.
    # Covers the max(1,.) clamp at the UF floor (4-channel stages) and the
    # decoder_type/use_strided wiring added for Stage 1b.
    torch.manual_seed(0)

    filter_configs = {
        "E1": (20, 72, 144, 72, 20),
        "U2": (20, 36, 72, 36, 12),
        "U4": (20, 20, 36, 20, 8),
        "U8": (20, 12, 20, 12, 4),
        "U16": (20, 8, 12, 8, 4),
        "UF": (20, 4, 4, 4, 4),
    }
    bottleneck_configs = {
        "enet_native": (4, 8, 8, 2, 1),
        "5": (4, 5, 5, 2, 1),
        "3": (4, 3, 3, 2, 1),
        "2": (4, 2, 2, 2, 1),
    }
    op_flag_combos = [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ]

    def symmetric(channels: tuple[int, int, int, int, int]) -> bool:
        # max_unpool needs MaxUnpool2d's indices channel count to match the
        # decoder stage they're applied at: initial==stage5, stage1==stage4.
        # Only E1/enet_paper-style symmetric configs satisfy this; the
        # Stage-2 filter axis (U2..UF) intentionally does not (f5 != f_i for
        # every row past E1) -- that axis is only valid under
        # upsample_conv, which is exactly why Stage 1b fixes the decoder
        # before Stage 2's grid runs.
        return channels[0] == channels[4] and channels[1] == channels[3]

    dummy = torch.zeros(1, 1, 512, 512)
    n_tested = 0
    n_skipped_asymmetric_max_unpool = 0
    for filter_name, channels in filter_configs.items():
        for bneck_name, bnecks in bottleneck_configs.items():
            for decoder_type in ("max_unpool", "upsample_conv"):
                if decoder_type == "max_unpool" and not symmetric(channels):
                    n_skipped_asymmetric_max_unpool += 1
                    continue
                for use_dilated, use_asymmetric, use_strided in op_flag_combos:
                    model = ENet(
                        in_channels=1,
                        out_channels=2,
                        channels=channels,
                        bottlenecks_per_stage=bnecks,
                        decoder_type=decoder_type,
                        use_dilated=use_dilated,
                        use_asymmetric=use_asymmetric,
                        use_strided=use_strided,
                    ).eval()
                    with torch.no_grad():
                        out = model(dummy)
                    assert out.shape == (1, 2, 512, 512), (
                        f"{filter_name}/{bneck_name}/{decoder_type}/"
                        f"d{use_dilated}a{use_asymmetric}s{use_strided}: got {tuple(out.shape)}"
                    )
                    n_tested += 1
    print(f"ENet self-test PASSED: {n_tested} configs built and forward-passed at 512x512.")
    print(
        f"({n_skipped_asymmetric_max_unpool} max_unpool combos skipped: asymmetric "
        "filter config, only valid under upsample_conv.)"
    )

    # 6-tuple split-stage2/stage3 self-test (upscale/'s combinatorial sweep):
    # confirms proj2_to_3 handles both directions (grow and shrink between
    # stage2 and stage3), and that the split is orthogonal to max_unpool's
    # f_i==f5/f1==f4 constraint -- only stage1/initial widths matter there,
    # never stage2/stage3 (see __init__'s comment on the 6-tuple branch).
    split_configs = {
        "split_grow_max_unpool": (20, 72, 96, 144, 72, 20),   # f_i=f5, f1=f4, f2<f3
        "split_shrink_max_unpool": (20, 72, 144, 96, 72, 20),  # f_i=f5, f1=f4, f2>f3
        "split_asymmetric_upsample_conv": (20, 36, 72, 144, 96, 12),  # fully asymmetric
    }
    n_split_tested = 0
    for name, channels6 in split_configs.items():
        decoder_type = "max_unpool" if "max_unpool" in name else "upsample_conv"
        model = ENet(
            in_channels=1, out_channels=2, channels=channels6,
            bottlenecks_per_stage=(4, 8, 8, 2, 1), decoder_type=decoder_type,
        ).eval()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 2, 512, 512), f"{name}: got {tuple(out.shape)}"
        assert not isinstance(model.proj2_to_3, torch.nn.Identity), f"{name}: expected a real projection, got Identity"
        n_split_tested += 1
    print(f"ENet 6-tuple split-stage2/3 self-test PASSED: {n_split_tested} configs.")

    # Sparse dilation pattern self-test (section 2a's div2/div4 bottleneck
    # axis): div2 = stage2 AND stage3 both at depth 4 (regular/dilated4/
    # regular/dilated16 each); div4 = stage3 REMOVED entirely (depth 0),
    # stage2 alone carries the same 4-op pattern. use_asymmetric=0 for both
    # (the grid's global op-flag choice), but the sparse pattern never emits
    # an asymmetric slot regardless, so this also checks that path is inert.
    sparse_configs = {
        "div2": (4, 4, 4, 2, 1),
        "div4": (4, 4, 0, 2, 1),
    }
    n_sparse_tested = 0
    for name, bnecks in sparse_configs.items():
        model = ENet(
            in_channels=1, out_channels=2, channels=(16, 16, 32, 16, 4),
            bottlenecks_per_stage=bnecks, decoder_type="upsample_conv",
            use_dilated=True, use_asymmetric=False, use_strided=True,
            context_pattern="sparse",
        ).eval()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 2, 512, 512), f"{name}: got {tuple(out.shape)}"
        assert len(model.stage2) == bnecks[1], f"{name}: stage2 depth {len(model.stage2)} != {bnecks[1]}"
        assert len(model.stage3) == bnecks[2], f"{name}: stage3 depth {len(model.stage3)} != {bnecks[2]}"
        for stage in (model.stage2, model.stage3):
            for i, block in enumerate(stage):
                expected_dilation = (1, 4, 1, 16)[i % 4]
                actual_dilation = block.conv_bn_act[0].dilation[0]
                assert actual_dilation == expected_dilation, (
                    f"{name}: block {i} dilation {actual_dilation} != expected {expected_dilation}"
                )
        n_sparse_tested += 1
    print(f"ENet sparse-dilation-pattern self-test PASSED: {n_sparse_tested} configs (div2/div4).")
