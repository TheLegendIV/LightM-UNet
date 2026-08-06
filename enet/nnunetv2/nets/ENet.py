from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

DecoderType = Literal["max_unpool", "upsample_conv"]
ContextPattern = Literal[
    "default", "sparse", "dense_dilation", "dense_dilation_a", "dense_dilation_reg_interleaved",
    "dense_dilation_reg_trailing",
]

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

# "Denser dilation" probe: every context-stage slot dilated (the 2/4/8/16
# schedule repeated twice over 8 slots), replacing CONTEXT_STAGE_PATTERN's
# remaining plain/asymmetric slots with dilation too -- no plain, no
# asymmetric, at all. Max rate stays 16 (unchanged from CONTEXT_STAGE_
# PATTERN) -- at stage2/3's 1/8 resolution (64x64 for a 512x512 input), a
# 3x3 kernel at dilation=16 already spans 2*16+1=33px, ~52% of the feature
# map; going higher starts hitting mostly zero-padding for a typical
# interior pixel (diminishing to negative returns), so this probe tests
# COVERAGE DENSITY at the existing safe ceiling, not reach beyond it (see
# SHALLOW_DILATION_WIDE_PATTERN below for the reach-focused probe instead,
# at a coarser resolution where there's still headroom).
DENSE_DILATION_PATTERN: tuple[dict, ...] = (
    {"padding": 2, "dilation": 2},
    {"padding": 4, "dilation": 4},
    {"padding": 8, "dilation": 8},
    {"padding": 16, "dilation": 16},
    {"padding": 2, "dilation": 2},
    {"padding": 4, "dilation": 4},
    {"padding": 8, "dilation": 8},
    {"padding": 16, "dilation": 16},
)

# "Schedule A" from compression/gridding_investigation/: replaces DENSE_
# DILATION_PATTERN's (2,4,8,16) -- every consecutive pair of which shares a
# factor of 2 -- with the fully pairwise-coprime (1,5,7,17). Same sum (30),
# so identical theoretical receptive field (61x61) and identical params/
# MACs (dilation rate doesn't change kernel size or channel count, it's
# free) -- the ONLY thing that changes is which input pixels are actually
# reachable. The investigation's own impulse-response test found the
# current schedule reaches just 25.8% of its 61x61 receptive field (a
# textbook gridding checkerboard -- every layer's sample grid aligns with
# every other's, from the shared factor of 2), vs. 87.3% for this schedule
# (near-solid fill, no shared factors anywhere). See
# compression/gridding_investigation/gridding_investigation_summary.md for
# the full writeup (task1_structural_results.json has the exact numbers) --
# structurally confirmed as a real fix, empirically inconclusive on
# whether it's *this model's* dominant error driver, hence retraining with
# it directly rather than treating the investigation as final.
DENSE_DILATION_SCHEDULE_A_PATTERN: tuple[dict, ...] = (
    {"padding": 1, "dilation": 1},
    {"padding": 5, "dilation": 5},
    {"padding": 7, "dilation": 7},
    {"padding": 17, "dilation": 17},
    {"padding": 1, "dilation": 1},
    {"padding": 5, "dilation": 5},
    {"padding": 7, "dilation": 7},
    {"padding": 17, "dilation": 17},
)

# DENSE_DILATION_PATTERN's (2,4,8,16) schedule, but with a full (non-DSC,
# projected) RegularBottleneck inserted before AND after each 4-rate cycle
# -- reg, 2, 4, 8, 16, reg, 2, 4, 8, 16, reg (11 slots, the boundary reg
# between the two cycles shared rather than doubled). Only meaningful
# under dsc_no_projection=True (see _make_context_stage): that mode
# otherwise runs the ENTIRE stage as depthwise+pointwise DSCNoProjection-
# Bottlenecks, which mix channels only through a single 1x1 pointwise conv
# per block, at full channel width, with no bottleneck squeeze anywhere.
# A plain RegularBottleneck's reduce -> 3x3 -> expand does the channel
# mixing and the 3x3 spatial conv AS ONE joint operation (not factored
# into depthwise-then-pointwise), reintroduced periodically. Callers
# should widen bottlenecks_per_stage's stage2/stage3 entries to 11 (from
# the native 8) to fit one full reg-bookended double cycle without
# truncating the trailing reg -- e.g. ENET_BOTTLENECKS="4,11,11,2,1".
# {"reg_bottleneck": True} is a sentinel _make_context_stage's dsc_no_
# projection branch checks BEFORE reading dilation/asymmetric keys, since
# it selects a different bottleneck CLASS, not just different conv kwargs
# within DSCNoProjectionBottleneck.
DENSE_DILATION_REG_INTERLEAVED_PATTERN: tuple[dict, ...] = (
    {"reg_bottleneck": True},
    {"padding": 2, "dilation": 2},
    {"padding": 4, "dilation": 4},
    {"padding": 8, "dilation": 8},
    {"padding": 16, "dilation": 16},
    {"reg_bottleneck": True},
    {"padding": 2, "dilation": 2},
    {"padding": 4, "dilation": 4},
    {"padding": 8, "dilation": 8},
    {"padding": 16, "dilation": 16},
    {"reg_bottleneck": True},
)

# Same reg/dilation interleaving idea as DENSE_DILATION_REG_INTERLEAVED_PATTERN,
# but the reg-bookend TRAILS each dilation cycle instead of leading it: 2, 4,
# 8, 16, reg -- a 5-slot cycle repeating via i % len(pattern), so n_ops=10
# gives exactly two full (dilate-then-consolidate) cycles with no truncation
# (vs. the 11-slot lead-in/lead-out version's 3 reg-bookends). Rationale: the
# stage immediately after down2 (or down1, for stage3) already got a real
# joint spatial+channel operation from the downsampling bottleneck's own
# reduce/main conv, so opening stage2/3 with ANOTHER reg block may be
# redundant -- this variant only spends a reg block where the stage has no
# other source of a joint (non-factorized) operation nearby: right before
# the dilated context gets handed off (to proj2_to_3/stage3, or to up4).
# Unlike the reg_bottleneck slots elsewhere, this pattern is meant to run
# under dsc_no_projection=False (DSC WITH projection kept on the dilated
# slots -- see _make_context_stage's "plain" loop, which now also honors
# the reg_bottleneck sentinel to force a genuine non-DSC RegularBottleneck
# for that one slot, matching the sentinel's existing meaning under
# dsc_no_projection=True).
DENSE_DILATION_REG_TRAILING_PATTERN: tuple[dict, ...] = (
    {"padding": 2, "dilation": 2},
    {"padding": 4, "dilation": 4},
    {"padding": 8, "dilation": 8},
    {"padding": 16, "dilation": 16},
    {"reg_bottleneck": True},
)

# Stage 4.4.1's shallow-stage pattern: alternating regular/dilated(16), for
# stage1 and regular4 -- both stages currently have NO dilation mechanism at
# all (only stage2/3's context pattern does). Same max dilation rate (16) as
# CONTEXT_STAGE_PATTERN's own highest rung, per the probe's own spec.
SHALLOW_DILATION_PATTERN: tuple[dict, ...] = (
    {},
    {"padding": 16, "dilation": 16},
)

# "Push reach further" probe: same alternating regular/dilated structure as
# SHALLOW_DILATION_PATTERN, but at dilation=32 instead of 16. stage1 and
# regular4 both run at 1/4 resolution (128x128 for a 512x512 input) --
# roughly double stage2/3's 64x64, so a 3x3 kernel at dilation=32 (span
# 2*32+1=65px) covers ~51% of THAT map, the same relative aggressiveness
# dilation=16 already has at stage2/3's finer resolution -- i.e. this
# stays inside the same safe ceiling SHALLOW_DILATION_PATTERN's dilation=16
# already respects, just exercised at a coarser stage that has the spatial
# headroom for it.
SHALLOW_DILATION_WIDE_PATTERN: tuple[dict, ...] = (
    {},
    {"padding": 32, "dilation": 32},
)

# "Dense" variants of the two shallow patterns above: every slot dilated
# (single-entry tuple, so i % len(pattern) always selects it), no
# alternating plain slots left -- same rationale as DENSE_DILATION_PATTERN
# but for stage1/regular4 instead of stage2/3. Gated by a separate
# shallow_dilation_dense flag rather than folded into shallow_dilation/
# shallow_dilation_wide directly, so "alternating" vs. "dense" stays an
# independent knob from "which rate" (16 vs. 32), matching how
# context_pattern="dense_dilation" is independent from use_dilated.
SHALLOW_DILATION_DENSE_PATTERN: tuple[dict, ...] = (
    {"padding": 16, "dilation": 16},
)
SHALLOW_DILATION_WIDE_DENSE_PATTERN: tuple[dict, ...] = (
    {"padding": 32, "dilation": 32},
)


def _activation(channels: int, relu: bool) -> nn.Module:
    return nn.ReLU(inplace=True) if relu else nn.PReLU(channels)


def _dilatable_conv_block(
    internal_channels: int, kernel_size: int, padding: int, dilation: int,
    relu: bool, use_dsc: bool, separable_dilated: bool,
) -> nn.Sequential:
    """Shared inner-conv builder for one (possibly dilated, non-asymmetric)
    context slot: plain k x k, DSC-factored (depthwise + pointwise), or
    (k,1)+(1,k) separable-dilated -- the same three-way branch
    RegularBottleneck's own non-asymmetric conv uses. Used by
    MergedContextBottleneck for BOTH its op1 and op2 halves, so either half
    can independently be plain, dilated, DSC'd, or separable-dilated,
    depending on what pattern slot it was built from."""
    if use_dsc:
        return nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, groups=internal_channels, bias=False),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
        )
    if separable_dilated and dilation != 1:
        return nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1),
                      padding=(padding, 0), dilation=dilation, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size),
                      padding=(0, padding), dilation=dilation, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
        )
    return nn.Sequential(
        nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(internal_channels),
        _activation(internal_channels, relu),
    )


def _reduce_proj(in_channels: int, internal_channels: int, relu: bool, double: bool) -> nn.Sequential:
    """1x1 squeeze projection, in_channels -> internal_channels, ending in an
    activation. Stage 4.5's double=True inserts a second internal_channels
    -> internal_channels (1x1 conv, BN, activation) unit after the first --
    twice the ops in the same projection for zero extra spatial buffering
    (1x1 convs need no SWG line buffer in FINN, unlike the 3x3/5x5 spatial
    convs this leaves untouched)."""
    layers = [
        nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(internal_channels),
        _activation(internal_channels, relu),
    ]
    if double:
        layers += [
            nn.Conv2d(internal_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
        ]
    return nn.Sequential(*layers)


def _expand_proj(internal_channels: int, out_channels: int, relu: bool, double: bool) -> nn.Sequential:
    """1x1 expand projection, internal_channels -> out_channels, ending in a
    bare BN (no activation -- every caller applies its own out_act AFTER
    adding the residual, not here). Stage 4.5's double=True inserts a
    leading internal_channels -> internal_channels (1x1 conv, BN,
    activation) unit before the final projection, same rationale as
    _reduce_proj."""
    layers = []
    if double:
        layers += [
            nn.Conv2d(internal_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
        ]
    layers += [
        nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    return nn.Sequential(*layers)


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
        separable_dilated: bool = False,
        double_projections: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)
        activation = nn.ReLU if relu else nn.PReLU

        self.reduce = _reduce_proj(channels, internal_channels, relu, double_projections)

        if asymmetric:
            if use_dsc:
                raise ValueError(
                    "use_dsc is not defined for asymmetric bottlenecks -- asymmetric is "
                    "itself a factorization of the inner conv, not exercised in combination "
                    "with DSC by any planned experiment. Disable use_asymmetric to use DSC."
                )
            if separable_dilated and dilation != 1:
                # Only an actual conflict when this slot is ALSO dilated --
                # separable_dilated is a no-op (never reached below) for a
                # plain asymmetric slot at dilation=1, which is what every
                # CONTEXT_STAGE_PATTERN asymmetric entry actually is. Callers
                # (e.g. _make_context_stage) pass separable_dilated=True
                # uniformly across a whole stage without needing to know
                # per-slot whether it's asymmetric.
                raise ValueError(
                    "separable_dilated is not defined for asymmetric bottlenecks -- both are "
                    "factorizations of the inner conv into a (k,1)+(1,k) pair; asymmetric already "
                    "IS that factorization (at dilation=1). Disable use_asymmetric to use "
                    "separable_dilated."
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
        elif separable_dilated and dilation != 1:
            # Probe 4.4.2: factor the dilated k x k conv into a (k,1) + (1,k)
            # pair, same dilation rate on both passes -- same K^2 -> 2K MAC
            # reduction the asymmetric branch already gets at dilation=1,
            # applied here to the dilated slots instead. Padding on each pass
            # is dilation*(k-1)/2 = dilation for a 3-tap kernel, matching how
            # `padding` is already derived for the monolithic dilated conv
            # elsewhere in this module (CONTEXT_STAGE_PATTERN's own entries).
            self.conv = nn.Sequential(
                nn.Conv2d(
                    internal_channels, internal_channels, kernel_size=(kernel_size, 1),
                    padding=(padding, 0), dilation=dilation, bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(internal_channels) if not relu else activation(inplace=True),
                nn.Conv2d(
                    internal_channels, internal_channels, kernel_size=(1, kernel_size),
                    padding=(0, padding), dilation=dilation, bias=False,
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
        self.expand = _expand_proj(internal_channels, channels, relu, double_projections)
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
        double_projections: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, out_channels // internal_ratio)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        if use_strided:
            # Spatial downsampling folded into the reduce conv itself (single
            # stride-2 2x2 conv) -- the ENet-native / FINN-favoured path.
            # This one stays single even when double_projections=True (it's
            # doing the spatial stride, not a plain 1x1 projection -- doubling
            # it would duplicate the downsampling itself, not just add cheap
            # 1x1 capacity); the extra unit goes on the (always-1x1) `expand`
            # side instead.
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
                *_reduce_proj(in_channels, internal_channels, relu, double_projections),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            _activation(internal_channels, relu),
        )
        self.expand = _expand_proj(internal_channels, out_channels, relu, double_projections)
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        relu: bool = True,
        double_projections: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, in_channels // internal_ratio)
        # main_proj is NOT doubled by double_projections -- probe 4.5 scoped
        # this to the reduce/expand pair specifically (confirmed), not the
        # skip-path projection.
        self.main_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.reduce = _reduce_proj(in_channels, internal_channels, relu, double_projections)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True) if relu else nn.PReLU(internal_channels),
        )
        self.expand = _expand_proj(internal_channels, out_channels, relu, double_projections)
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


class DSCNoProjectionBottleneck(nn.Module):
    """DSC everywhere, without the bottleneck's own 1x1 reduce/expand
    projection pair: depthwise KxK (groups=channels, one filter per
    channel, dilation-aware) + pointwise 1x1 (channels -> channels, the
    ONLY channel-mixing step in the block) applied directly at full
    `channels` width, no squeeze down to internal_channels/expand back up
    at all. RegularBottleneck's use_dsc=True only swaps the INNER conv for
    this same depthwise+pointwise pair while keeping its own reduce/expand
    around it (see RegularBottleneck's `elif use_dsc:` branch); this class
    removes that surrounding structure entirely -- used network-wide
    (stage1/regular1, stage2, stage3, regular4, regular5) when
    ENet(dsc_no_projection=True). Not combinable with asymmetric slots
    (same constraint RegularBottleneck's own use_dsc already has -- callers
    must set use_asymmetric=0)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
        dropout_p: float = 0.1,
        relu: bool = False,
    ):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                      dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            activation(channels) if not relu else activation(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.out_act = activation(channels) if not relu else activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv(x))
        return self.out_act(x + out)


class MergedContextBottleneck(nn.Module):
    """Probe 4.4.3: fuses a PAIR of context-stage pattern slots into ONE
    block sharing a single reduce/expand projection pair -- reduce ->
    op1(first_kwargs) -> op2(dilation) -> expand -- instead of two separate
    blocks each with their own reduce/expand. Halves the block count in
    stage2/3 for the same op coverage. `first_kwargs` is normally one of
    CONTEXT_STAGE_PATTERN's own {} / {"kernel_size":5,"padding":2,
    "asymmetric":True} entries (the non-dilated half of the default
    regular-or-asymmetric/dilated pairing), but under
    context_pattern="dense_dilation" every slot is itself dilated, so
    `first_kwargs` may also carry its own "dilation" key -- op1 respects
    that the same way op2 always has, giving a (dilated, dilated) pair
    (probe 5.3) instead of (plain-or-asymmetric, dilated). use_dsc/
    separable_dilated apply independently to op1 (when non-asymmetric) and
    op2 via the shared _dilatable_conv_block helper."""

    def __init__(
        self,
        channels: int,
        first_kwargs: dict,
        dilation: int,
        internal_ratio: int = 4,
        dropout_p: float = 0.1,
        relu: bool = False,
        use_dsc: bool = False,
        separable_dilated: bool = False,
    ):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)
        self.reduce = _reduce_proj(channels, internal_channels, relu, double=False)

        kernel_size = first_kwargs.get("kernel_size", 3)
        padding = first_kwargs.get("padding", 1)
        dilation1 = first_kwargs.get("dilation", 1)
        if first_kwargs.get("asymmetric", False):
            self.op1 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1),
                          padding=(padding, 0), bias=False),
                nn.BatchNorm2d(internal_channels),
                _activation(internal_channels, relu),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size),
                          padding=(0, padding), bias=False),
                nn.BatchNorm2d(internal_channels),
                _activation(internal_channels, relu),
            )
        else:
            self.op1 = _dilatable_conv_block(
                internal_channels, kernel_size, padding, dilation1, relu, use_dsc, separable_dilated,
            )

        # Second op: normally the dilated 3x3 (the pairing's dilated half).
        self.op2 = _dilatable_conv_block(
            internal_channels, 3, dilation, dilation, relu, use_dsc, separable_dilated,
        )

        self.expand = _expand_proj(internal_channels, channels, relu, double=False)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.out_act = _activation(channels, relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.op1(out)
        out = self.op2(out)
        out = self.dropout(self.expand(out))
        return self.out_act(x + out)


class TwoBlockSkipStage(nn.Module):
    """Probe 4.6: wraps a list of same-width bottleneck ops (a context
    stage's op list) with an ADDITIONAL short residual on top of each op's
    own internal residual -- every 2 consecutive ops form a group, and the
    group's input is added to the group's output. Confined to one context
    stage (stage2 or stage3 individually; channels are constant throughout
    each, so no projection is needed for the shortcut). A trailing odd op
    (only possible if a stage's depth is odd) is left un-grouped."""

    def __init__(self, ops: list[nn.Module]):
        super().__init__()
        self.ops = nn.ModuleList(ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i, n = 0, len(self.ops)
        while i < n:
            group_input = x
            x = self.ops[i](x)
            if i + 1 < n:
                x = self.ops[i + 1](x)
                x = x + group_input
            i += 2
        return x


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
        shallow_dilation: bool = False,
        separable_dilated: bool = False,
        merge_dilated_pairs: bool = False,
        dsc_dilated_only: bool = False,
        double_projections: bool = False,
        two_block_skip: bool = False,
        dsc_no_projection: bool = False,
        shallow_dilation_wide: bool = False,
        shallow_dilation_dense: bool = False,
        dsc_no_projection_context_only: bool = False,
        reg_bookend_dsc: bool = False,
    ):
        super().__init__()
        valid_context_patterns = (
            "default", "sparse", "dense_dilation", "dense_dilation_a", "dense_dilation_reg_interleaved",
            "dense_dilation_reg_trailing",
        )
        if context_pattern not in valid_context_patterns:
            raise ValueError(f"context_pattern must be one of {valid_context_patterns}, got {context_pattern!r}.")
        if shallow_dilation_dense and not (shallow_dilation or shallow_dilation_wide):
            raise ValueError(
                "shallow_dilation_dense modifies shallow_dilation/shallow_dilation_wide's alternating "
                "pattern into an every-slot-dilated one -- meaningless without one of them also set."
            )
        if dsc_no_projection_context_only and not dsc_no_projection:
            raise ValueError(
                "dsc_no_projection_context_only narrows dsc_no_projection's scope to stage2/stage3 only "
                "(regular1/regular4/regular5 revert to normal projected bottlenecks) -- meaningless "
                "without dsc_no_projection=True itself."
            )
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
        # Stage 4 architecture probes (all default False/off -- every prior
        # config, including every Stage 1/1b/1c/2/2a/2b/3 sweep cell, builds
        # byte-identical to before these were added):
        self.shallow_dilation = shallow_dilation          # 4.4.1
        self.separable_dilated = separable_dilated        # 4.4.2
        self.merge_dilated_pairs = merge_dilated_pairs    # 4.4.3
        self.dsc_dilated_only = dsc_dilated_only          # 4.4.4
        self.double_projections = double_projections      # 4.5
        self.two_block_skip = two_block_skip              # 4.6
        self.dsc_no_projection = dsc_no_projection         # DSC everywhere, no reduce/expand
        self.shallow_dilation_wide = shallow_dilation_wide  # push shallow-stage dilation reach to 32
        self.shallow_dilation_dense = shallow_dilation_dense  # every shallow-stage slot dilated, not alternating
        # dsc_no_projection scoped to stage2/stage3 (where the dilated
        # slots are) only -- regular1/regular4/regular5 build normally
        # (projected, non-DSC unless the global use_dsc is also set).
        # Isolates whether the no-projection savings specifically come
        # from the dilated bottlenecks, or from applying it network-wide.
        self.dsc_no_projection_context_only = dsc_no_projection_context_only
        # Applies DSC (WITH its projection kept) to context_pattern=
        # "dense_dilation_reg_interleaved"'s reg-bookend slots -- those
        # currently use a full-rank (non-DSC) RegularBottleneck; this
        # tests whether the same depthwise-separable factoring that (with
        # NO projection) helps the dilated slots also saves compute/memory
        # on the non-dilated bookends once projection is kept. No-op
        # unless context_pattern="dense_dilation_reg_interleaved" and
        # dsc_no_projection=True (the only path that ever builds a
        # reg-bookend slot).
        self.reg_bookend_dsc = reg_bookend_dsc

        n_stage1, n_stage2, n_stage3, n_regular4, n_regular5 = self.bottlenecks_per_stage

        self.initial = self._build_initial_block(in_channels, initial_channels)
        self.down1 = DownsamplingBottleneck(
            initial_channels, stage1_channels, dropout_p=0.01, use_strided=use_strided,
            relu=not use_prelu, double_projections=double_projections,
        )
        self.regular1 = self._make_shallow_stage(stage1_channels, n_stage1, dropout_p=0.01, relu=not use_prelu)

        self.down2 = DownsamplingBottleneck(
            stage1_channels, stage2_channels, dropout_p=0.1, use_strided=use_strided,
            relu=not use_prelu, double_projections=double_projections,
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

        self.up4 = UpsamplingBottleneck(stage3_channels, stage4_channels, double_projections=double_projections)
        self.regular4 = self._make_shallow_stage(stage4_channels, n_regular4, dropout_p=0.1, relu=True)
        self.up5 = UpsamplingBottleneck(stage4_channels, stage5_channels, double_projections=double_projections)
        if dsc_no_projection and not dsc_no_projection_context_only:
            self.regular5 = nn.Sequential(
                *[DSCNoProjectionBottleneck(stage5_channels, dropout_p=0.1, relu=True) for _ in range(n_regular5)]
            )
        else:
            self.regular5 = nn.Sequential(
                *[RegularBottleneck(stage5_channels, dropout_p=0.1, relu=True, use_dsc=use_dsc,
                                     double_projections=double_projections)
                  for _ in range(n_regular5)]
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

    def _make_context_stage(self, channels: int, n_ops: int) -> nn.Module:
        """Builds an n_ops-long context stage from the first n_ops entries of
        CONTEXT_STAGE_PATTERN (or SPARSE_DILATION_PATTERN when
        context_pattern="sparse" -- section 2a's reduced-depth bottleneck
        axis, div2/div4: regular/dilated4/regular/dilated16, no 2/8 rungs,
        never asymmetric; or DENSE_DILATION_PATTERN when
        context_pattern="dense_dilation" -- every slot dilated, no plain/
        asymmetric slots at all, same 2/4/8/16 max-rate ceiling repeated
        twice over 8 slots). use_dilated/use_asymmetric downgrade the
        matching slots to a plain regular bottleneck rather than dropping
        them, so stage depth and op composition stay independent knobs
        (Stage 2 sweeps depth; Stage 1b's op ablation sweeps composition).
        use_dsc factorizes whatever inner conv results (plain or dilated --
        not asymmetric, RegularBottleneck rejects that combination).

        Also: DENSE_DILATION_SCHEDULE_A_PATTERN when context_pattern=
        "dense_dilation_a" (same shape as dense_dilation but the gridding-
        investigation's coprime (1,5,7,17) rates instead of (2,4,8,16));
        DENSE_DILATION_REG_INTERLEAVED_PATTERN when context_pattern=
        "dense_dilation_reg_interleaved" (11 slots: reg,2,4,8,16,reg,2,4,8,
        16,reg -- only meaningful under dsc_no_projection, see that
        pattern's own module comment); DENSE_DILATION_REG_TRAILING_PATTERN
        when context_pattern="dense_dilation_reg_trailing" (5-slot cycle:
        2,4,8,16,reg -- reg trails each dilation cycle instead of leading
        it, meant for dsc_no_projection=False so use_dsc keeps the dilated
        slots' own reduce/expand projection; see that pattern's own module
        comment). Both reg-bookend patterns rely on the SAME
        {"reg_bottleneck": True} sentinel, honored in both the
        dsc_no_projection branch below AND the plain (projected) loop at
        the bottom of this method.

        Stage 4 probes layer on top of this same pattern:
        merge_dilated_pairs (4.4.3) pairs up (non-dilated, dilated) slots
        into one MergedContextBottleneck each, halving the block count;
        separable_dilated (4.4.2) and dsc_dilated_only (4.4.4) modify how a
        dilated slot's inner conv is built; two_block_skip (4.6) wraps the
        finished op list in an extra short residual. dsc_no_projection
        (DSC everywhere, no reduce/expand) takes priority over all of the
        above -- every slot becomes a DSCNoProjectionBottleneck instead,
        keeping only the pattern's dilation schedule (except
        {"reg_bottleneck": True} slots, which become a full RegularBottleneck)."""
        if self.context_pattern == "sparse":
            pattern = SPARSE_DILATION_PATTERN
        elif self.context_pattern == "dense_dilation":
            pattern = DENSE_DILATION_PATTERN
        elif self.context_pattern == "dense_dilation_a":
            pattern = DENSE_DILATION_SCHEDULE_A_PATTERN
        elif self.context_pattern == "dense_dilation_reg_interleaved":
            pattern = DENSE_DILATION_REG_INTERLEAVED_PATTERN
        elif self.context_pattern == "dense_dilation_reg_trailing":
            pattern = DENSE_DILATION_REG_TRAILING_PATTERN
        else:
            pattern = CONTEXT_STAGE_PATTERN

        if self.dsc_no_projection:
            ops = []
            for i in range(n_ops):
                kwargs = dict(pattern[i % len(pattern)])
                if kwargs.get("reg_bottleneck", False):
                    # DENSE_DILATION_REG_INTERLEAVED_PATTERN's bookend slots:
                    # a RegularBottleneck (real reduce/expand projection),
                    # not a DSCNoProjectionBottleneck -- a deliberately
                    # DIFFERENT bottleneck class per slot, not just
                    # different conv kwargs. use_dsc=False by default (a
                    # full-rank 3x3 that mixes channels and space jointly
                    # in one op, instead of DSC's depthwise-then-pointwise
                    # factorization); reg_bookend_dsc=True switches this to
                    # DSC WITH the projection kept, testing whether the
                    # same factoring that (with NO projection) helps the
                    # dilated slots also pays off here once projection is
                    # retained.
                    ops.append(RegularBottleneck(
                        channels, dropout_p=0.1, use_dsc=self.reg_bookend_dsc, relu=not self.use_prelu,
                        double_projections=self.double_projections,
                    ))
                    continue
                if kwargs.get("asymmetric", False):
                    if self.use_asymmetric:
                        raise ValueError(
                            "dsc_no_projection is not defined for asymmetric bottlenecks -- set "
                            "use_asymmetric=0 (same constraint RegularBottleneck's own use_dsc already has)."
                        )
                    kwargs = {}
                dilation = kwargs.get("dilation", 1)
                if dilation != 1 and not self.use_dilated:
                    dilation = 1
                ops.append(DSCNoProjectionBottleneck(
                    channels, kernel_size=3, padding=dilation, dilation=dilation,
                    dropout_p=0.1, relu=not self.use_prelu,
                ))
            return TwoBlockSkipStage(ops) if self.two_block_skip else nn.Sequential(*ops)

        if self.merge_dilated_pairs:
            ops = []
            i = 0
            while i < n_ops:
                first_kwargs = dict(pattern[i % len(pattern)])
                if first_kwargs.get("asymmetric", False) and not self.use_asymmetric:
                    first_kwargs = {}
                # Under context_pattern="dense_dilation" (probe 5.3), the
                # "first" half of the pair is itself dilated (pattern has no
                # more plain/asymmetric slots) -- downgrade it the same way
                # a dilated slot is downgraded everywhere else in this
                # module when use_dilated is off.
                if first_kwargs.get("dilation", 1) != 1 and not self.use_dilated:
                    first_kwargs = {}
                if i + 1 < n_ops:
                    dilation = dict(pattern[(i + 1) % len(pattern)]).get("dilation", 1)
                    if dilation != 1 and not self.use_dilated:
                        dilation = 1
                    ops.append(MergedContextBottleneck(
                        channels, first_kwargs, dilation, dropout_p=0.1,
                        relu=not self.use_prelu, use_dsc=self.use_dsc,
                        separable_dilated=self.separable_dilated,
                    ))
                    i += 2
                else:
                    # Odd leftover slot (only possible if n_ops is odd) --
                    # falls back to one plain RegularBottleneck for it.
                    ops.append(RegularBottleneck(channels, dropout_p=0.1, use_dsc=self.use_dsc,
                                                  relu=not self.use_prelu,
                                                  double_projections=self.double_projections, **first_kwargs))
                    i += 1
            return TwoBlockSkipStage(ops) if self.two_block_skip else nn.Sequential(*ops)

        ops = []
        for i in range(n_ops):
            kwargs = dict(pattern[i % len(pattern)])
            # DENSE_DILATION_REG_TRAILING_PATTERN's reg-bookend slot -- forces
            # a genuine non-DSC RegularBottleneck here regardless of the
            # global use_dsc, same meaning the sentinel already has under
            # dsc_no_projection=True (see that branch above). Every OTHER
            # existing pattern never sets this key, so this is a no-op for
            # them (kwargs.pop returns False, nothing changes).
            is_reg_bookend = kwargs.pop("reg_bottleneck", False)
            is_dilated = kwargs.get("dilation", 1) != 1
            if is_dilated and not self.use_dilated:
                kwargs = {}
                is_dilated = False
            if kwargs.get("asymmetric", False) and not self.use_asymmetric:
                kwargs = {}
            if is_reg_bookend:
                use_dsc_here = False
            else:
                use_dsc_here = self.use_dsc or (self.dsc_dilated_only and is_dilated)
            ops.append(RegularBottleneck(
                channels, dropout_p=0.1, use_dsc=use_dsc_here, relu=not self.use_prelu,
                separable_dilated=self.separable_dilated, double_projections=self.double_projections,
                **kwargs,
            ))
        return TwoBlockSkipStage(ops) if self.two_block_skip else nn.Sequential(*ops)

    def _make_shallow_stage(self, channels: int, n_ops: int, dropout_p: float, relu: bool) -> nn.Sequential:
        """Builds stage1 (regular1) or regular4 -- both plain (undilated)
        RegularBottleneck repeats by default, same as before Stage 4.
        shallow_dilation=True (probe 4.4.1) instead alternates
        [regular, dilated(16)] via SHALLOW_DILATION_PATTERN;
        shallow_dilation_wide=True does the same but at dilation=32 via
        SHALLOW_DILATION_WIDE_PATTERN (stage1/regular4 run at 1/4
        resolution, coarser than stage2/3's 1/8, so there's headroom for a
        wider rate there -- see that pattern's own module-level comment for
        the receptive-field math). shallow_dilation_dense=True (probe 5.2)
        swaps whichever of those two patterns is selected for its "dense"
        counterpart (every slot dilated, no alternating plain slot) --
        independent of which rate (16 vs. 32) is in use. All three respect
        use_dilated the same way _make_context_stage's pattern slots do
        (downgrade to plain regular if use_dilated is off). regular5 never
        calls this -- probe 4.4.1 explicitly excludes it (built directly in
        __init__, unchanged). dsc_no_projection (DSC everywhere, no
        reduce/expand) takes priority over all dilation variants here --
        plain (undilated) DSCNoProjectionBottleneck repeats, matching
        regular5's own dsc_no_projection handling in __init__ -- UNLESS
        dsc_no_projection_context_only is also set, in which case this
        stage ignores dsc_no_projection entirely and builds normally (it's
        not stage2/stage3, so it's out of that flag's narrowed scope)."""
        if self.dsc_no_projection and not self.dsc_no_projection_context_only:
            return nn.Sequential(
                *[DSCNoProjectionBottleneck(channels, dropout_p=dropout_p, relu=relu) for _ in range(n_ops)]
            )
        if self.shallow_dilation_wide:
            pattern = SHALLOW_DILATION_WIDE_DENSE_PATTERN if self.shallow_dilation_dense else SHALLOW_DILATION_WIDE_PATTERN
        elif self.shallow_dilation:
            pattern = SHALLOW_DILATION_DENSE_PATTERN if self.shallow_dilation_dense else SHALLOW_DILATION_PATTERN
        else:
            return nn.Sequential(
                *[RegularBottleneck(channels, dropout_p=dropout_p, use_dsc=self.use_dsc, relu=relu,
                                     double_projections=self.double_projections)
                  for _ in range(n_ops)]
            )
        ops = []
        for i in range(n_ops):
            kwargs = dict(pattern[i % len(pattern)])
            if kwargs.get("dilation", 1) != 1 and not self.use_dilated:
                kwargs = {}
            ops.append(RegularBottleneck(channels, dropout_p=dropout_p, use_dsc=self.use_dsc, relu=relu,
                                          double_projections=self.double_projections, **kwargs))
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

    # Stage 4 architecture-probe self-test: each of the 6 new flags,
    # individually, at U4 (4,16,32,16,4) / native bottlenecks (4,8,8,2,1) --
    # the shared reference every stage_4_*.job probe is built on.
    U4_CHANNELS = (4, 16, 32, 16, 4)
    U4_BOTTLENECKS = (4, 8, 8, 2, 1)
    stage4_configs = {
        "4.4.1_shallow_dilation": dict(shallow_dilation=True),
        "4.4.2_separable_dilated": dict(separable_dilated=True),
        "4.4.3_merge_dilated_pairs": dict(merge_dilated_pairs=True),
        "4.4.4_dsc_dilated_only": dict(dsc_dilated_only=True),
        "4.5_double_projections": dict(double_projections=True),
        "4.6_two_block_skip": dict(two_block_skip=True),
        "4.8_dense_dilation": dict(context_pattern="dense_dilation"),
        "4.9_shallow_dilation_wide": dict(shallow_dilation_wide=True),
    }
    n_stage4_tested = 0
    for name, kwargs in stage4_configs.items():
        model = ENet(
            in_channels=1, out_channels=5, channels=U4_CHANNELS,
            bottlenecks_per_stage=U4_BOTTLENECKS, decoder_type="upsample_conv",
            use_prelu=False, **kwargs,
        ).eval()
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 5, 512, 512), f"{name}: got {tuple(out.shape)}"
        n_stage4_tested += 1
    # merge_dilated_pairs halves stage2/3's block count (8 -> 4 at native depth).
    merged = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS,
        bottlenecks_per_stage=U4_BOTTLENECKS, decoder_type="upsample_conv",
        use_prelu=False, merge_dilated_pairs=True,
    )
    assert len(merged.stage2) == 4, f"merge_dilated_pairs: stage2 has {len(merged.stage2)} blocks, expected 4"
    assert len(merged.stage3) == 4, f"merge_dilated_pairs: stage3 has {len(merged.stage3)} blocks, expected 4"
    # two_block_skip wraps stage2/3 in TwoBlockSkipStage but preserves depth.
    skipped = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS,
        bottlenecks_per_stage=U4_BOTTLENECKS, decoder_type="upsample_conv",
        use_prelu=False, two_block_skip=True,
    )
    assert isinstance(skipped.stage2, TwoBlockSkipStage), "two_block_skip: stage2 not wrapped"
    assert len(skipped.stage2) == 8 and len(skipped.stage3) == 8, "two_block_skip: unexpected depth"

    # dsc_no_projection: DSC everywhere, no reduce/expand -- needs
    # use_asymmetric=0 (asymmetric slots are undefined for it, same
    # constraint use_dsc already has), unlike the shared loop above.
    no_proj = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS,
        bottlenecks_per_stage=U4_BOTTLENECKS, decoder_type="upsample_conv",
        use_prelu=False, use_asymmetric=False, dsc_no_projection=True,
    ).eval()
    with torch.no_grad():
        out = no_proj(dummy)
    assert out.shape == (1, 5, 512, 512), f"dsc_no_projection: got {tuple(out.shape)}"
    for stage in (no_proj.regular1, no_proj.stage2, no_proj.stage3, no_proj.regular4, no_proj.regular5):
        for block in stage:
            assert isinstance(block, DSCNoProjectionBottleneck), f"dsc_no_projection: found {type(block).__name__}"
    # Asymmetric slots must be rejected outright, not silently downgraded.
    try:
        ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
             decoder_type="upsample_conv", use_prelu=False, dsc_no_projection=True)
        raise AssertionError("dsc_no_projection: expected ValueError with use_asymmetric=1 (default), got none")
    except ValueError:
        pass
    n_stage4_tested += 1

    # dense_dilation: every stage2/3 slot dilated (no plain, no asymmetric),
    # 2/4/8/16 repeated twice -- confirms DENSE_DILATION_PATTERN's schedule
    # actually reached every block, not just that it built without error.
    dense = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS,
        bottlenecks_per_stage=U4_BOTTLENECKS, decoder_type="upsample_conv",
        use_prelu=False, context_pattern="dense_dilation",
    )
    expected_dilations = (2, 4, 8, 16, 2, 4, 8, 16)
    for stage in (dense.stage2, dense.stage3):
        for i, block in enumerate(stage):
            actual = block.conv_bn_act[0].dilation[0]
            assert actual == expected_dilations[i], (
                f"dense_dilation: block {i} dilation {actual} != {expected_dilations[i]}"
            )

    # shallow_dilation_wide: stage1/regular4 alternate regular/dilated(32).
    wide = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS,
        bottlenecks_per_stage=U4_BOTTLENECKS, decoder_type="upsample_conv",
        use_prelu=False, shallow_dilation_wide=True,
    )
    for stage_name, stage in (("regular1", wide.regular1), ("regular4", wide.regular4)):
        for i, block in enumerate(stage):
            expected = 32 if i % 2 == 1 else 1
            actual = block.conv_bn_act[0].dilation[0]
            assert actual == expected, f"shallow_dilation_wide: {stage_name}[{i}] dilation {actual} != {expected}"

    print(f"ENet Stage-4 architecture-probe self-test PASSED: {n_stage4_tested} flags, individually, at U4.")

    # Stage 5 architecture-probe-PAIRS self-test: combinations of two
    # stage_4 flags at once, off the same U4 reference. Structural checks
    # go beyond "it builds" where a pair could plausibly silently degrade
    # into just one flag's behavior (e.g. dilation not actually reaching
    # op1 of a merged pair, or "dense" not actually reaching every slot).
    E1_CHANNELS = (4, 16, 28, 16, 4)

    def _dilation_of(seq: nn.Sequential) -> int:
        return seq[0].dilation[0]

    # 5.1 dsc_no_projection + dense_dilation: every block DSC'd (no
    # projection) AND stage2/3's DSC depthwise conv follows the dense
    # 2/4/8/16 x2 schedule (not just plain dsc_no_projection's dilation=1).
    pair1 = ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 dsc_no_projection=True, context_pattern="dense_dilation")
    with torch.no_grad():
        assert pair1(dummy).shape == (1, 5, 512, 512)
    for stage in (pair1.regular1, pair1.stage2, pair1.stage3, pair1.regular4, pair1.regular5):
        for block in stage:
            assert isinstance(block, DSCNoProjectionBottleneck)
    for i, block in enumerate(pair1.stage2):
        expected = (2, 4, 8, 16, 2, 4, 8, 16)[i]
        assert _dilation_of(block.conv) == expected, f"5.1: stage2[{i}] dilation {_dilation_of(block.conv)} != {expected}"

    # 5.2 dense_dilation + shallow_dilation_wide(dense): stage1/regular4
    # EVERY slot at dilation=32 (not alternating), stage2/3 dense schedule.
    pair2 = ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 context_pattern="dense_dilation", shallow_dilation_wide=True, shallow_dilation_dense=True)
    with torch.no_grad():
        assert pair2(dummy).shape == (1, 5, 512, 512)
    for stage_name, stage in (("regular1", pair2.regular1), ("regular4", pair2.regular4)):
        for i, block in enumerate(stage):
            actual = _dilation_of(block.conv_bn_act)
            assert actual == 32, f"5.2: {stage_name}[{i}] dilation {actual} != 32 (dense -- every slot, not alternating)"

    # 5.3 dense_dilation + merge_dilated_pairs: stage2/3 halved to 4 blocks,
    # and BOTH op1 and op2 of each merged block carry real (different)
    # dilation rates -- (2,4), (8,16), (2,4), (8,16) -- not op1 silently
    # staying plain (the pre-fix behavior first_kwargs' dilation key was
    # ignored for).
    pair3 = ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 context_pattern="dense_dilation", merge_dilated_pairs=True)
    with torch.no_grad():
        assert pair3(dummy).shape == (1, 5, 512, 512)
    assert len(pair3.stage2) == 4 and len(pair3.stage3) == 4, "5.3: expected 4 merged blocks per stage"
    expected_pairs = ((2, 4), (8, 16), (2, 4), (8, 16))
    for stage in (pair3.stage2, pair3.stage3):
        for i, block in enumerate(stage):
            d1, d2 = _dilation_of(block.op1), _dilation_of(block.op2)
            assert (d1, d2) == expected_pairs[i], f"5.3: block {i} dilations {(d1, d2)} != {expected_pairs[i]}"

    # 5.4 dense_dilation + E1Shape: just channels + dense schedule compose.
    pair4 = ENet(in_channels=1, out_channels=5, channels=E1_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 context_pattern="dense_dilation")
    with torch.no_grad():
        assert pair4(dummy).shape == (1, 5, 512, 512)
    assert pair4.stage2[0].reduce[0].out_channels == 7, "5.4: expected E1Shape's stage2/3 width (28) // 4 internal"

    # 5.5 dsc_no_projection + E1Shape.
    pair5 = ENet(in_channels=1, out_channels=5, channels=E1_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False, dsc_no_projection=True)
    with torch.no_grad():
        assert pair5(dummy).shape == (1, 5, 512, 512)

    # 5.6 separable_dilated + dense_dilation: every one of stage2/3's 8
    # slots is now dilated (dense), so separable_dilated's (k,1)+(1,k)
    # factoring should apply to ALL 8, not just the original pattern's 4.
    pair6 = ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 context_pattern="dense_dilation", separable_dilated=True)
    with torch.no_grad():
        assert pair6(dummy).shape == (1, 5, 512, 512)
    for stage in (pair6.stage2, pair6.stage3):
        for i, block in enumerate(stage):
            # RegularBottleneck's own separable_dilated branch: [Conv2d(k,1),
            # BN, act, Conv2d(1,k)] -- 4 modules; the final BN+act comes from
            # the block's own outer conv_bn_act wrapper, not duplicated here
            # (unlike MergedContextBottleneck's op1/op2, which have no such
            # outer wrapper and so are each fully self-contained -- 6
            # modules when separable-factored, see the 5.7 check below).
            assert isinstance(block.conv, nn.Sequential) and len(block.conv) == 4, (
                f"5.6: block {i} not separable-factored (expected a 4-module (k,1)+(1,k) Sequential, "
                f"got {type(block.conv).__name__} len={len(block.conv) if hasattr(block.conv, '__len__') else 'n/a'})"
            )

    # 5.7 merge_dilated_pairs + separable_dilated: halved block count, and
    # op2 (the dilated half of the default regular-or-asymmetric/dilated
    # pairing) is separable-factored; op1 (dilation=1 here, default
    # pattern) stays a plain 3-module block since separable_dilated is a
    # no-op at dilation=1.
    pair7 = ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 merge_dilated_pairs=True, separable_dilated=True)
    with torch.no_grad():
        assert pair7(dummy).shape == (1, 5, 512, 512)
    assert len(pair7.stage2) == 4 and len(pair7.stage3) == 4, "5.7: expected 4 merged blocks per stage"
    for stage in (pair7.stage2, pair7.stage3):
        for block in stage:
            assert len(block.op2) == 6, "5.7: op2 (dilated half) not separable-factored"
            assert len(block.op1) == 3, "5.7: op1 (dilation=1 half) unexpectedly factored"

    # 5.8 shallow_dilation_wide + merge_dilated_pairs: orthogonal stages
    # (stage1/4 vs. stage2/3) -- both effects present simultaneously.
    pair8 = ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
                 decoder_type="upsample_conv", use_prelu=False, use_asymmetric=False,
                 shallow_dilation_wide=True, merge_dilated_pairs=True)
    with torch.no_grad():
        assert pair8(dummy).shape == (1, 5, 512, 512)
    assert len(pair8.stage2) == 4 and len(pair8.stage3) == 4, "5.8: expected 4 merged blocks per stage"
    for stage_name, stage in (("regular1", pair8.regular1), ("regular4", pair8.regular4)):
        for i, block in enumerate(stage):
            expected = 32 if i % 2 == 1 else 1
            actual = _dilation_of(block.conv_bn_act)
            assert actual == expected, f"5.8: {stage_name}[{i}] dilation {actual} != {expected}"

    print("ENet Stage-5 architecture-probe-PAIRS self-test PASSED: 8 pairs.")

    # Stage 6 -- DscNoProjDense variants (compression/gridding_investigation/'s
    # follow-up): dense_dilation_a swaps in the coprime (1,5,7,17) schedule;
    # dense_dilation_reg_interleaved bookends each (2,4,8,16) cycle with a
    # full RegularBottleneck, at bumped stage2/3 depth (11, not the native 8).
    schedule_a = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        dsc_no_projection=True, context_pattern="dense_dilation_a",
    )
    with torch.no_grad():
        assert schedule_a(dummy).shape == (1, 5, 512, 512)
    expected_a_dilations = (1, 5, 7, 17, 1, 5, 7, 17)
    for stage in (schedule_a.stage2, schedule_a.stage3):
        for i, block in enumerate(stage):
            assert isinstance(block, DSCNoProjectionBottleneck), f"6.1: block {i} not DSC'd"
            actual = _dilation_of(block.conv)
            assert actual == expected_a_dilations[i], f"6.1: block {i} dilation {actual} != {expected_a_dilations[i]}"

    reg_interleaved = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=(4, 11, 11, 2, 1),
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        dsc_no_projection=True, context_pattern="dense_dilation_reg_interleaved",
    )
    with torch.no_grad():
        assert reg_interleaved(dummy).shape == (1, 5, 512, 512)
    expected_types = [RegularBottleneck, DSCNoProjectionBottleneck, DSCNoProjectionBottleneck,
                       DSCNoProjectionBottleneck, DSCNoProjectionBottleneck, RegularBottleneck,
                       DSCNoProjectionBottleneck, DSCNoProjectionBottleneck, DSCNoProjectionBottleneck,
                       DSCNoProjectionBottleneck, RegularBottleneck]
    for stage_name, stage in (("stage2", reg_interleaved.stage2), ("stage3", reg_interleaved.stage3)):
        assert len(stage) == 11, f"6.2: {stage_name} has {len(stage)} blocks, expected 11"
        for i, (block, expected_type) in enumerate(zip(stage, expected_types)):
            assert type(block) is expected_type, (
                f"6.2: {stage_name}[{i}] is {type(block).__name__}, expected {expected_type.__name__}"
            )
    # reg_bottleneck slots must have a real reduce/expand projection (unlike
    # their DSC neighbors, which have none at all).
    assert hasattr(reg_interleaved.stage2[0], "reduce"), "6.2: reg_bottleneck slot missing its projection"
    assert not hasattr(reg_interleaved.stage2[1], "reduce"), "6.2: DSC slot unexpectedly has a projection"

    print("ENet Stage-6 DscNoProjDense-variant self-test PASSED: 2 configs (schedule_a, reg_interleaved).")

    # Stage 8 -- RegInterleaved variants isolating where dsc_no_projection's
    # savings actually come from: 8.1 narrows its scope to stage2/3 only
    # (regular1/regular4/regular5 revert to normal projected bottlenecks);
    # 8.3 applies DSC (with projection kept) to the reg-bookend slots
    # themselves, instead of leaving them full-rank.
    REG_INTERLEAVED_BOTTLENECKS = (4, 11, 11, 2, 1)

    # 8.1 dsc_no_projection_context_only: regular1/regular4/regular5 must
    # be plain RegularBottleneck (projected, non-DSC since global use_dsc
    # is off) -- NOT DSCNoProjectionBottleneck -- while stage2/3 keep the
    # exact same mixed reg-bookend/DSC-no-projection structure as plain
    # reg_interleaved (unaffected by this flag, since it's already scoped
    # to stage2/3 only).
    context_only = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=REG_INTERLEAVED_BOTTLENECKS,
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        dsc_no_projection=True, dsc_no_projection_context_only=True,
        context_pattern="dense_dilation_reg_interleaved",
    )
    with torch.no_grad():
        assert context_only(dummy).shape == (1, 5, 512, 512)
    for stage_name, stage in (("regular1", context_only.regular1), ("regular5", context_only.regular5)):
        for i, block in enumerate(stage):
            assert type(block) is RegularBottleneck, (
                f"8.1: {stage_name}[{i}] is {type(block).__name__}, expected RegularBottleneck (context-only "
                "should keep dsc_no_projection out of the shallow stages)"
            )
            assert hasattr(block, "reduce"), f"8.1: {stage_name}[{i}] missing its reduce/expand projection"
    for i, (block, expected_type) in enumerate(zip(context_only.stage2, expected_types)):
        assert type(block) is expected_type, (
            f"8.1: stage2[{i}] is {type(block).__name__}, expected {expected_type.__name__} "
            "(stage2/3 should be unaffected by context_only, same as plain reg_interleaved)"
        )
    # Narrowing the scope should never cost more params than the unscoped
    # version -- regular1/4/5 reverting to projected bottlenecks is
    # strictly cheaper for these channel widths (DSCNoProjectionBottleneck
    # operates at full width with no squeeze, which is more expensive than
    # a projected reduce->3x3->expand at this scale).
    context_only_params = sum(p.numel() for p in context_only.parameters())
    reg_interleaved_params = sum(p.numel() for p in reg_interleaved.parameters())
    assert context_only_params < reg_interleaved_params, (
        f"8.1: context-only params ({context_only_params}) should be less than "
        f"unscoped reg_interleaved's ({reg_interleaved_params})"
    )
    # Without dsc_no_projection itself, the context_only flag is meaningless
    # and must be rejected outright, not silently ignored.
    try:
        ENet(in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=REG_INTERLEAVED_BOTTLENECKS,
             decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
             dsc_no_projection_context_only=True, context_pattern="dense_dilation_reg_interleaved")
        raise AssertionError("8.1: expected ValueError with dsc_no_projection=False (default), got none")
    except ValueError:
        pass

    # 8.3 reg_bookend_dsc: the reg-bookend slots (stage2[0]/stage2[5]/
    # stage2[10] per DENSE_DILATION_REG_INTERLEAVED_PATTERN) must become
    # DSC'd (a Sequential depthwise+pointwise conv, not a bare Conv2d) but
    # KEEP their reduce/expand projection -- unlike the dilated slots,
    # which have no projection at all.
    reg_bookend_dsc_model = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=REG_INTERLEAVED_BOTTLENECKS,
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        dsc_no_projection=True, reg_bookend_dsc=True, context_pattern="dense_dilation_reg_interleaved",
    )
    with torch.no_grad():
        assert reg_bookend_dsc_model(dummy).shape == (1, 5, 512, 512)
    for stage in (reg_bookend_dsc_model.stage2, reg_bookend_dsc_model.stage3):
        for i, block in enumerate(stage):
            if i in (0, 5, 10):  # reg-bookend slots
                assert type(block) is RegularBottleneck, f"8.3: block {i} is {type(block).__name__}, expected RegularBottleneck"
                assert hasattr(block, "reduce"), f"8.3: reg-bookend block {i} missing its projection"
                assert isinstance(block.conv, nn.Sequential), f"8.3: reg-bookend block {i}'s conv not DSC-factored"
            else:  # dilated slots, unaffected by reg_bookend_dsc
                assert type(block) is DSCNoProjectionBottleneck, f"8.3: block {i} is {type(block).__name__}, expected DSCNoProjectionBottleneck"
    reg_bookend_dsc_params = sum(p.numel() for p in reg_bookend_dsc_model.parameters())
    assert reg_bookend_dsc_params < reg_interleaved_params, (
        f"8.3: reg_bookend_dsc params ({reg_bookend_dsc_params}) should be less than "
        f"the full-rank bookend version's ({reg_interleaved_params})"
    )

    # 8.2 (plain ReLU instead of PReLU) is just use_prelu=False composed
    # with the existing reg_interleaved config -- no new flag, already
    # covered by every use_prelu=False build elsewhere in this self-test.

    print("ENet Stage-8 RegInterleaved-isolation self-test PASSED: 2 new flags (dsc_no_projection_context_only, reg_bookend_dsc).")

    # Stage 9 -- DSC-WITH-projection dense_dilation variants (probing the
    # buffer-memory analysis in compression/: does dsc_no_projection's full-
    # channel-width DSC cost accuracy, or just memory?). All three keep
    # RegularBottleneck's own reduce/expand (use_dsc=True only swaps the
    # INNER conv for depthwise+pointwise, still at internal_channels) --
    # unlike every dsc_no_projection=True config above, which drops that
    # projection entirely.
    def _dsc_dilation_of(block: RegularBottleneck) -> int:
        # block.conv is a Sequential(depthwise, pointwise) when use_dsc=True
        # (see RegularBottleneck's `elif use_dsc:` branch) -- conv[0] is the
        # depthwise conv itself, unlike _dilation_of's conv_bn_act[0] path
        # (which only works for a bare, non-DSC'd inner conv).
        return block.conv[0].dilation[0]

    # 9.1 dense_dilation + DSC + projection, native depth (8 slots, all
    # dilated 2,4,8,16 x2, no reg-bookend at all).
    dsc_projected = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=U4_BOTTLENECKS,
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        use_dsc=True, context_pattern="dense_dilation",
    )
    with torch.no_grad():
        assert dsc_projected(dummy).shape == (1, 5, 512, 512)
    expected_9_1_dilations = (2, 4, 8, 16, 2, 4, 8, 16)
    for stage_name, stage in (("stage2", dsc_projected.stage2), ("stage3", dsc_projected.stage3)):
        assert len(stage) == 8, f"9.1: {stage_name} has {len(stage)} blocks, expected 8"
        for i, block in enumerate(stage):
            assert type(block) is RegularBottleneck, f"9.1: {stage_name}[{i}] is {type(block).__name__}, expected RegularBottleneck"
            assert hasattr(block, "reduce"), f"9.1: {stage_name}[{i}] missing its reduce/expand projection (DSC should not drop it here)"
            assert isinstance(block.conv, nn.Sequential), f"9.1: {stage_name}[{i}]'s conv not DSC-factored"
            actual = _dsc_dilation_of(block)
            assert actual == expected_9_1_dilations[i], f"9.1: {stage_name}[{i}] dilation {actual} != {expected_9_1_dilations[i]}"
    # regular1/regular4/regular5 also inherit use_dsc=True (dsc_no_projection
    # is off, so _make_shallow_stage's plain branch applies self.use_dsc) --
    # but MUST still keep their own reduce/expand, unlike dsc_no_projection's
    # regular1/4/5 (a DSCNoProjectionBottleneck, no projection at all).
    for stage_name, stage in (("regular1", dsc_projected.regular1), ("regular5", dsc_projected.regular5)):
        for i, block in enumerate(stage):
            assert type(block) is RegularBottleneck, f"9.1: {stage_name}[{i}] is {type(block).__name__}, expected RegularBottleneck"
            assert hasattr(block, "reduce"), f"9.1: {stage_name}[{i}] missing its reduce/expand projection"
            assert isinstance(block.conv, nn.Sequential), f"9.1: {stage_name}[{i}]'s conv not DSC-factored"

    # 9.2 same as 9.1 but stage2/3 widened to 11 blocks (matching
    # RegInterleaved's total depth) -- isolates whether extra DEPTH alone
    # (not the reg-bookend's specific non-DSC character) explains any of
    # RegInterleaved's edge over the native-depth version. DENSE_DILATION_
    # PATTERN is 8 slots long, so slot 8/9/10 wrap back to its own start
    # (2,4,8 again) via the existing i % len(pattern) repeat -- no new code,
    # purely bottlenecks_per_stage=11 composed with 9.1's flags.
    dsc_projected_deep = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=(4, 11, 11, 2, 1),
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        use_dsc=True, context_pattern="dense_dilation",
    )
    with torch.no_grad():
        assert dsc_projected_deep(dummy).shape == (1, 5, 512, 512)
    expected_9_2_dilations = (2, 4, 8, 16, 2, 4, 8, 16, 2, 4, 8)
    for stage_name, stage in (("stage2", dsc_projected_deep.stage2), ("stage3", dsc_projected_deep.stage3)):
        assert len(stage) == 11, f"9.2: {stage_name} has {len(stage)} blocks, expected 11"
        for i, block in enumerate(stage):
            assert type(block) is RegularBottleneck, f"9.2: {stage_name}[{i}] is {type(block).__name__}, expected RegularBottleneck"
            actual = _dsc_dilation_of(block)
            assert actual == expected_9_2_dilations[i], f"9.2: {stage_name}[{i}] dilation {actual} != {expected_9_2_dilations[i]}"
    dsc_projected_params = sum(p.numel() for p in dsc_projected.parameters())
    dsc_projected_deep_params = sum(p.numel() for p in dsc_projected_deep.parameters())
    assert dsc_projected_deep_params > dsc_projected_params, (
        f"9.2: deepened params ({dsc_projected_deep_params}) should exceed the native-depth version's "
        f"({dsc_projected_params}) -- 3 extra blocks per stage must add real parameters"
    )

    # 9.3 dense_dilation_reg_trailing: 2,4,8,16,reg x2 (10 slots) -- the
    # reg-bookend TRAILS each dilation cycle instead of leading it (unlike
    # dense_dilation_reg_interleaved's reg-first pattern), and -- unlike
    # 9.1/9.2's dilated slots -- the reg slot itself must be forced non-DSC
    # (a bare Conv2d, not a Sequential) even though the global use_dsc=True,
    # via the {"reg_bottleneck": True} sentinel now also honored in the
    # "plain" (projected) loop, not just the dsc_no_projection branch.
    reg_trailing = ENet(
        in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=(4, 10, 10, 2, 1),
        decoder_type="upsample_conv", use_prelu=True, use_asymmetric=False,
        use_dsc=True, context_pattern="dense_dilation_reg_trailing",
    )
    with torch.no_grad():
        assert reg_trailing(dummy).shape == (1, 5, 512, 512)
    reg_slot_indices = (4, 9)
    expected_9_3_dilations = {0: 2, 1: 4, 2: 8, 3: 16, 5: 2, 6: 4, 7: 8, 8: 16}
    for stage_name, stage in (("stage2", reg_trailing.stage2), ("stage3", reg_trailing.stage3)):
        assert len(stage) == 10, f"9.3: {stage_name} has {len(stage)} blocks, expected 10"
        for i, block in enumerate(stage):
            assert type(block) is RegularBottleneck, f"9.3: {stage_name}[{i}] is {type(block).__name__}, expected RegularBottleneck"
            assert hasattr(block, "reduce"), f"9.3: {stage_name}[{i}] missing its reduce/expand projection"
            if i in reg_slot_indices:
                assert isinstance(block.conv, torch.nn.Conv2d), (
                    f"9.3: {stage_name}[{i}] (reg-trailing slot) is DSC-factored -- expected a bare, "
                    "non-DSC Conv2d (the reg_bottleneck sentinel must force use_dsc=False here)"
                )
                assert block.conv.dilation[0] == 1, f"9.3: {stage_name}[{i}] reg slot dilation != 1"
            else:
                assert isinstance(block.conv, nn.Sequential), f"9.3: {stage_name}[{i}] (dilated slot) not DSC-factored"
                actual = _dsc_dilation_of(block)
                assert actual == expected_9_3_dilations[i], f"9.3: {stage_name}[{i}] dilation {actual} != {expected_9_3_dilations[i]}"

    print("ENet Stage-9 DSC+projection dense_dilation self-test PASSED: 3 configs (dsc_projected, dsc_projected_deep, reg_trailing) + 1 new context_pattern (dense_dilation_reg_trailing).")

    # Stage 10 -- "breeding" probe: dense_dilation_reg_interleaved's proven
    # bookend structure (stage 6/8) crossed with separable_dilated applied
    # to a DENSE conv (stage 5's real quadratic-term win, NOT the depthwise
    # version stage 9.5 dropped), with the projection kept throughout
    # (dsc_no_projection=False, use_dsc=False) -- no new code, every piece
    # of this combination already exists individually; this is the first
    # time they're composed together. The reg-bookend slots naturally stay
    # plain dense 3x3 (dilation=1 skips separable_dilated's own `dilation
    # != 1` guard) without needing special-casing.
    for relu_flag, label in ((False, "10.1 (prelu)"), (True, "10.2 (relu)")):
        hybrid = ENet(
            in_channels=1, out_channels=5, channels=U4_CHANNELS, bottlenecks_per_stage=(4, 11, 11, 2, 1),
            decoder_type="upsample_conv", use_prelu=not relu_flag, use_asymmetric=False,
            use_dsc=False, separable_dilated=True, dsc_no_projection=False,
            context_pattern="dense_dilation_reg_interleaved",
        )
        with torch.no_grad():
            assert hybrid(dummy).shape == (1, 5, 512, 512)
        for stage_name, stage in (("stage2", hybrid.stage2), ("stage3", hybrid.stage3)):
            assert len(stage) == 11, f"{label}: {stage_name} has {len(stage)} blocks, expected 11"
            for i, block in enumerate(stage):
                assert type(block) is RegularBottleneck, f"{label}: {stage_name}[{i}] is {type(block).__name__}, expected RegularBottleneck"
                assert hasattr(block, "reduce"), f"{label}: {stage_name}[{i}] missing its reduce/expand projection"
                conv = block.conv
                if i in (0, 5, 10):  # reg-bookend slots
                    assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (3, 3), (
                        f"{label}: {stage_name}[{i}] (bookend) is {type(conv).__name__}, expected a plain dense 3x3 Conv2d"
                    )
                    assert conv.groups == 1, f"{label}: {stage_name}[{i}] (bookend) unexpectedly grouped/depthwise"
                else:  # dilated slots -- dense (k,1)+(1,k), NOT depthwise
                    assert isinstance(conv, nn.Sequential) and conv[0].kernel_size == (3, 1), (
                        f"{label}: {stage_name}[{i}] (dilated) is {type(conv).__name__}, expected a separable-factored Sequential starting (3,1)"
                    )
                    assert conv[0].groups == 1, f"{label}: {stage_name}[{i}] (dilated) unexpectedly depthwise -- should stay dense like S5-SeparableDense, not DSC"

    print("ENet Stage-10 reg_interleaved+separable_dilated hybrid self-test PASSED: 2 configs (prelu, relu), 0 new code -- pure composition of existing flags.")
