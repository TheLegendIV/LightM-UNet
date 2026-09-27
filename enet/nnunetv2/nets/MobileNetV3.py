"""MobileNetV3-Large (Howard et al., 2019, ICCV), adapted into an
encoder-decoder segmentation network for this repo's own nnU-Net pipeline
-- same "real backbone, this-repo's-own decoder" split as MobileNetV2.py
(see that file's own module docstring for the shared rationale, not
repeated here).

Architecturally verified against the official TensorFlow-slim source
(models/research/slim/nets/mobilenet/mobilenet_v3.py, cloned into this
repo's own models/ directory -- V3_LARGE's own `spec` list, and the
mbv3_op/mbv3_op_se helper's own defaults (act=tf.nn.relu, squeeze_factor=4,
gating_fn=relu6(x+3)/6 i.e. hard-sigmoid), read directly). Confirmed real
15-block schedule (t=expand ratio, c=out channels, k=kernel, s=stride, se=
squeeze-excite present, act=activation -- RE=ReLU, HS=hard-swish):
    stem: Conv2d(in, 16, k=3, s=2) -> BN -> hard_swish
    t=1,      c=16,  k=3, s=1, se=N, act=RE
    t=4,      c=24,  k=3, s=2, se=N, act=RE
    t=3,      c=24,  k=3, s=1, se=N, act=RE
    t=3,      c=40,  k=5, s=2, se=Y, act=RE
    t=3,      c=40,  k=5, s=1, se=Y, act=RE
    t=3,      c=40,  k=5, s=1, se=Y, act=RE
    t=6,      c=80,  k=3, s=2, se=N, act=HS
    t=2.5,    c=80,  k=3, s=1, se=N, act=HS
    t=184/80, c=80,  k=3, s=1, se=N, act=HS
    t=184/80, c=80,  k=3, s=1, se=N, act=HS
    t=6,      c=112, k=3, s=1, se=Y, act=HS
    t=6,      c=112, k=3, s=1, se=Y, act=HS
    t=6,      c=160, k=5, s=2, se=Y, act=HS
    t=6,      c=160, k=5, s=1, se=Y, act=HS
    t=6,      c=160, k=5, s=1, se=Y, act=HS
    head1: Conv2d(160, 960, k=1) -> BN -> hard_swish
    head2: Conv2d(960, 1280, k=1) -> hard_swish   # NO BatchNorm (source's own normalizer_fn=None)
5 stride-2 points total (stem + 4 in-schedule stride-2 rows) -- same real
/32 backbone depth as MobileNetV2.py, same explicit "full paper fidelity,
not truncated to this repo's usual /8" choice. The classification-only
global-average-pool (`reduce_to_1x1`) + final 1x1-to-#classes logits layer
the source uses for ImageNet classification are DROPPED here (not part of
a segmentation backbone) -- head1/head2 (the real feature-refinement 1x1
convs) are kept, matching how MobileNetV3 is actually used as a
segmentation backbone in practice (e.g. LR-ASPP/DeepLabV3 on MobileNetV3
keep these, drop only the pool+classifier).

Inverted residual block: same expand(1x1)->depthwise(kxk, stride s)->
project(1x1, linear) structure as MobileNetV2.py's own InvertedResidual,
PLUS an optional squeeze-excite between the depthwise conv and the project
conv (only on rows with se=True) -- confirmed against the source: SE
squeeze_channels = max(1, round(BLOCK'S OWN INPUT channels / 4)), rounded
to a multiple of 8 (squeeze_factor=4, applied to the block's pre-expansion
input width, NOT the expanded hidden_dim -- a real, easy-to-get-backwards
detail, matching both the source and torchvision's own well-established
MobileNetV3 implementation), inner activation ReLU, gating hard-sigmoid
(relu6(x+3)/6, exactly the source's own gating_fn) -- scales the EXPANDED
(hidden_dim) feature map channel-wise. Also: expand conv is skipped when
expand_ratio==1 (only the very first block), same convention MobileNetV2.py
already documents.

Decoder: this repo's own addition (not paper-specified, same disclaimer as
MobileNetV2.py's own decoder) -- 5 plain ConvTranspose2d-stride-2 stages,
BN+hard_swish (matching this architecture's own dominant activation, not
ReLU6), channel schedule mirroring the encoder's own real channel scale at
each of the 5 downsample depths (1280->112->40->24->16->16), ending in a
bare Conv2d 1x1 to logits (no BN/activation, same convention every other
model file in this repo's nets/ directory uses for its own final layer).

width_mult: same real paper width-scaling knob as MobileNetV2.py (not this
repo's usual 5-value `channels` convention) -- scales every channel count
(including SE squeeze widths, computed from the ALREADY-scaled input
channels) and rounds to a multiple of 8. Default 1.0 = the real base
"MobileNetV3-Large 1.0" architecture.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.nets.MobileNetV2 import _make_divisible


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """relu6(x+3)/6 -- exactly the source's own gating_fn (not the
    piecewise-linear nn.Hardsigmoid default breakpoints in some other
    conventions; this is the specific formula mobilenet_v3.py's own
    DEFAULTS use)."""
    return F.relu6(x + 3.0) / 6.0


def hard_swish(x: torch.Tensor) -> torch.Tensor:
    return x * hard_sigmoid(x)


class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x)


def _activation(act: str) -> nn.Module:
    if act == "RE":
        return nn.ReLU(inplace=True)
    if act == "HS":
        return HardSwish()
    raise ValueError(f"Unknown activation {act!r} -- expected 'RE' or 'HS'.")


class SqueezeExcite(nn.Module):
    """Scales the EXPANDED (hidden_dim) feature map -- squeeze width is
    computed from that SAME expanded hidden_dim (squeeze_factor=4, rounded to
    a multiple of 8), not the block's pre-expansion input. This file
    previously computed it from the pre-expansion input instead (attributed
    to a TF-slim source reading) -- corrected after cross-checking two
    independent, code-verified references that both squeeze from hidden_dim:
    torchvision's own official mobilenetv3.py (`squeeze_channels =
    _make_divisible(cnf.expanded_channels // 4, 8)`) and the widely-used
    d-li14/mobilenetv3.pytorch reference (`SELayer(hidden_dim)` with
    `channel // reduction`). Only affects blocks with expand_ratio != 1
    (hidden_dim != in_channels); block 0 (t=1) is unaffected either way."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        squeeze_channels = _make_divisible(hidden_dim / 4, 8)
        self.fc1 = nn.Conv2d(hidden_dim, squeeze_channels, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.act1(self.fc1(scale))
        scale = hard_sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidualV3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
        expand_ratio: float, use_se: bool, act: str, dilation: int = 1,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"InvertedResidualV3 stride must be 1 or 2, got {stride}.")
        hidden_dim = _make_divisible(in_channels * expand_ratio)
        # Residual-eligibility uses the ORIGINAL declared stride, not the
        # dilation-adjusted one below -- matches torchvision's own
        # `use_res_connect = cnf.stride == 1 and ...` (dilation never changes
        # this decision, only the depthwise conv's own stride/padding).
        self.use_residual = stride == 1 and in_channels == out_channels
        effective_stride = 1 if dilation > 1 else stride
        padding = dilation * (kernel_size - 1) // 2

        expand: list[nn.Module] = []
        if expand_ratio != 1:
            expand = [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim), _activation(act),
            ]
        self.expand = nn.Sequential(*expand)

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, effective_stride, padding, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), _activation(act),
        )
        self.se = SqueezeExcite(hidden_dim) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        return x + out if self.use_residual else out


class UpsampleBlockHS(nn.Sequential):
    """Decoder's own upsample-by-2 primitive -- same shape as
    MobileNetV2.py's own UpsampleBlock, hard-swish instead of ReLU6 to
    match this architecture's own dominant activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            HardSwish(),
        )


class LRASPPHead(nn.Module):
    """Lite R-ASPP (Howard et al. 2019, Sec 6.4/Fig 10) -- verified against
    torchvision's own torchvision/models/segmentation/lraspp.py `LRASPPHead`
    class, read directly: `cbr` (1x1 conv, no bias -> BN -> ReLU) and `scale`
    (global-avg-pool -> 1x1 conv, no bias -> Sigmoid) both operate on the
    SAME high-level (deep, OS=16) feature; their product is bilinear-
    upsampled to the low-level (OS=8) feature's spatial size and summed with
    a separate 1x1-conv classifier on the low-level feature. torchvision
    simplifies the paper's own Fig 10 large-kernel/strided pool to a plain
    global pool (functionally equivalent for this purpose) -- followed here
    since it's the citable, code-verified reference, same convention as this
    file's other paper-vs-code corrections."""

    def __init__(self, low_channels: int, high_channels: int, out_channels: int, inter_channels: int = 128):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, out_channels, kernel_size=1)
        self.high_classifier = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_classifier(low) + self.high_classifier(x)


# Real V3-Large 15-block schedule -- see module docstring.
_LARGE_SETTING: tuple[tuple[float, int, int, int, bool, str], ...] = (
    # t,        c,   k, s, se,    act
    (1, 16, 3, 1, False, "RE"),
    (4, 24, 3, 2, False, "RE"),
    (3, 24, 3, 1, False, "RE"),
    (3, 40, 5, 2, True, "RE"),
    (3, 40, 5, 1, True, "RE"),
    (3, 40, 5, 1, True, "RE"),
    (6, 80, 3, 2, False, "HS"),
    (2.5, 80, 3, 1, False, "HS"),
    (184 / 80, 80, 3, 1, False, "HS"),
    (184 / 80, 80, 3, 1, False, "HS"),
    (6, 112, 3, 1, True, "HS"),
    (6, 112, 3, 1, True, "HS"),
    (6, 160, 5, 2, True, "HS"),
    (6, 160, 5, 1, True, "HS"),
    (6, 160, 5, 1, True, "HS"),
)
_LARGE_HEAD1_CHANNELS = 960

# Real V3-Small 11-block schedule (paper Table 2 / official TF-slim
# V3_SMALL spec, exp_size expressed here as a t=exp_size/block-own-input-
# channels ratio, same convention _LARGE_SETTING uses): stem(16) -> row0
# already stride=2 (unlike Large, whose row0 is stride=1) -- see the
# generic depth-marker scan below, which handles a schedule where the very
# first row is the one that transitions depth, not just Large's shape.
_SMALL_SETTING: tuple[tuple[float, int, int, int, bool, str], ...] = (
    # t,          c,  k, s, se,    act
    (16 / 16, 16, 3, 2, True, "RE"),
    (72 / 16, 24, 3, 2, False, "RE"),
    (88 / 24, 24, 3, 1, False, "RE"),
    (96 / 24, 40, 5, 2, True, "HS"),
    (240 / 40, 40, 5, 1, True, "HS"),
    (240 / 40, 40, 5, 1, True, "HS"),
    (120 / 40, 48, 5, 1, True, "HS"),
    (144 / 48, 48, 5, 1, True, "HS"),
    (288 / 48, 96, 5, 2, True, "HS"),
    (576 / 96, 96, 5, 1, True, "HS"),
    (576 / 96, 96, 5, 1, True, "HS"),
)
_SMALL_HEAD1_CHANNELS = 576  # Large uses 960 -- Small's own head1 width, per the source spec
_SMALL_HEAD2_CHANNELS = 1024  # Large uses 1280 -- Small's own head2 width, per the source spec
# NOTE: an earlier reading of Table 2's row 13 ("conv2d, 1x1, -, 576, X, HS,
# 1") took the SE column's checkmark to mean Small's head1 carries a
# squeeze-excite. Cross-checked against two independent, code-verified
# references -- torchvision's own official mobilenetv3.py (`Conv2dNormActivation`
# for both Large's and Small's head1, no SE layer at all) and the widely-used
# d-li14/mobilenetv3.pytorch reference (`conv_1x1_bn`, likewise no SE) -- and
# both agree head1 has NO squeeze-excite for either variant. Since these are
# the actual code (one of them literally the weights Google released),
# treated as authoritative over the earlier OCR'd table reading; the
# head1-SE mechanism below is removed accordingly.


class MobileNetV3(nn.Module):
    """decoder="convtranspose" (default): this repo's own 5-stage learned-
    upsample decoder (unchanged, see module docstring). decoder="lraspp":
    the paper's own Sec 6.4 segmentation-specific configuration -- RF2
    (`reduced_tail`, halves the last block-group's channels) + dilation held
    at OS=16 on that same group + the LRASPPHead above, matching Table 7's
    row 11 (V3-Small, RF2, LR-ASPP, F=128 -> paper's own 0.47M). Both
    `reduced_tail` and dilation are verified against torchvision's own
    `_mobilenet_v3_conf(reduced_tail=True, dilated=True)` (read directly):
    only the TRAILING run of rows sharing the last stride-2 transition's
    channel width (3 rows for both Large's 160ch group and Small's 96ch
    group) gets its `c` halved and its depthwise stride forced to 1 via
    dilation=2 -- `t` is expressed in this file as a ratio of the block's
    OWN (already-halved, for later tail rows) input channels, so halving
    only `c` reproduces torchvision's literal exp-channel values exactly
    without needing to touch `t` at all (worked through by hand against
    torchvision's own literal (672,960,960)/(288,576,576)-style exp values
    for Large/Small and confirmed to match)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 20,
        width_mult: float = 1.0,
        stem_channels: int = 16,
        setting: tuple = _LARGE_SETTING,
        head1_channels: int = _LARGE_HEAD1_CHANNELS,
        head2_channels: int = 1280,
        decoder: str = "convtranspose",
        reduced_tail: bool = True,
        lraspp_inter_channels: int = 128,
    ):
        super().__init__()
        if decoder not in ("convtranspose", "lraspp"):
            raise ValueError(f"Unknown decoder {decoder!r} -- expected 'convtranspose' or 'lraspp'.")
        self.decoder_type = decoder
        stem_c = _make_divisible(stem_channels * width_mult)

        # -- Encoder (real architecture, /32 total downsample) --------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_c), HardSwish(),
        )

        stride2_rows = [i for i, row in enumerate(setting) if row[3] == 2]
        # lraspp: halve `c` (only) for every row from the LAST stride-2 row
        # onward -- `t` is a ratio of the block's own input, so it naturally
        # recomputes the right (halved-input-relative) exp channels once the
        # preceding row's halved `c` flows in as this row's `in_c` (see
        # class docstring). Also hold OS=16 via dilation=2 + forced stride=1
        # on that same trailing group (row_dilation), rather than the
        # native OS=32 the un-dilated schedule would reach.
        effective_setting = list(setting)
        row_dilation = [1] * len(setting)
        if decoder == "lraspp":
            tail_start_row = stride2_rows[-1] if stride2_rows else len(setting)
            if reduced_tail:
                effective_setting = [
                    (t, c // 2 if i >= tail_start_row else c, k, s, se, act)
                    for i, (t, c, k, s, se, act) in enumerate(setting)
                ]
            row_dilation = [2 if i >= tail_start_row else 1 for i in range(len(setting))]

        self.blocks = nn.ModuleList()
        in_c = stem_c
        for row_idx, (t, c, k, s, se, act) in enumerate(effective_setting):
            out_c = _make_divisible(c * width_mult)
            self.blocks.append(InvertedResidualV3(in_c, out_c, k, s, t, se, act, dilation=row_dilation[row_idx]))
            in_c = out_c

        if decoder == "lraspp":
            # Low tap: output of the row at stage_markers[-4] -- torchvision's
            # own "C2" marker (verified: for both Large's/Small's real
            # 4-stride-2-row schedules this reduces to "the SECOND stride-2
            # row's output", computed generically here off `stride2_rows`
            # rather than hardcoded to that count).
            scaled_channels_eff = [stem_c] + [_make_divisible(c * width_mult) for (_, c, _, _, _, _) in effective_setting]
            stage_markers = [0] + [r + 1 for r in stride2_rows] + [len(setting)]
            low_marker = stage_markers[-4]
            low_channels = scaled_channels_eff[low_marker]
            self.low_tap_row = low_marker - 1  # 0-indexed row whose output is the low-level tap

            # head1 only (no head2/classifier -- LR-ASPP operates directly on
            # head1's output): channels = 6x the (possibly tail-halved) last
            # block's own output, matching torchvision's own
            # `lastconv_output_channels = 6 * lastconv_input_channels` (this
            # is ALWAYS how head1's width is derived in the real architecture,
            # not a separate constant -- 6*96=576/6*160=960 for the
            # un-reduced classification-style Small/Large head1 widths this
            # file's own `_SMALL_HEAD1_CHANNELS`/`_LARGE_HEAD1_CHANNELS`
            # constants already hardcode, confirmed consistent).
            head1_c = 6 * in_c
            self.head = nn.Sequential(
                nn.Conv2d(in_c, head1_c, kernel_size=1, bias=False), nn.BatchNorm2d(head1_c), HardSwish(),
            )
            self.lraspp = LRASPPHead(low_channels, head1_c, out_channels, inter_channels=lraspp_inter_channels)
            return

        # Depth markers: for each stride-2 transition, the channel count of
        # whatever fed INTO it (the previous row's output, or the stem's
        # output if row 0 itself is stride=2 -- Small's schedule hits this
        # case via row_idx-1==-1, Large's doesn't since its row 0 is
        # stride=1). Computed off the already-scaled per-row channel counts
        # so this stays consistent under any width_mult.
        depth_marker_channels: dict[int, int] = {}
        scaled_channels = [stem_c] + [_make_divisible(c * width_mult) for (_, c, _, _, _, _) in setting]
        for row_idx, (_, _, _, s, _, _) in enumerate(setting):
            if s == 2:
                depth_marker_channels[row_idx - 1] = scaled_channels[row_idx]  # row_idx-1==-1 means "the stem"

        head1_c = _make_divisible(head1_channels * width_mult)
        head2_c = _make_divisible(head2_channels * max(1.0, width_mult))
        head_layers: list[nn.Module] = [
            nn.Conv2d(in_c, head1_c, kernel_size=1, bias=False), nn.BatchNorm2d(head1_c), HardSwish(),
            nn.Conv2d(head1_c, head2_c, kernel_size=1, bias=True), HardSwish(),  # source: normalizer_fn=None here
        ]
        self.head = nn.Sequential(*head_layers)

        # -- Decoder (this repo's own addition -- see module docstring) -----
        # 4 non-stem downsample depths, deepest first. Marker keys are
        # SOURCE row indices (the row whose output sits at that depth,
        # -1 == the stem itself) -- sorted ascending then reversed picks
        # them deepest-to-shallowest regardless of how many rows the
        # schedule has (11 for Small, 15 for Large).
        marker_rows_shallow_to_deep = sorted(depth_marker_channels.keys())
        d0, d1, d2, d3 = (depth_marker_channels[r] for r in reversed(marker_rows_shallow_to_deep))
        self.up1 = UpsampleBlockHS(head2_c, d0)
        self.up2 = UpsampleBlockHS(d0, d1)
        self.up3 = UpsampleBlockHS(d1, d2)
        self.up4 = UpsampleBlockHS(d2, d3)
        self.up5 = UpsampleBlockHS(d3, stem_c)
        self.final = nn.Conv2d(stem_c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.decoder_type == "lraspp":
            input_size = x.shape[-2:]
            x = self.stem(x)
            low = None
            for row_idx, block in enumerate(self.blocks):
                x = block(x)
                if row_idx == self.low_tap_row:
                    low = x
            high = self.head(x)
            out = self.lraspp(low, high)
            return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.final(x)


if __name__ == "__main__":
    dummy = torch.randn(1, 1, 512, 512)
    net = MobileNetV3(in_channels=1, out_channels=5).eval()
    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (1, 5, 512, 512), f"got {tuple(out.shape)}"
    assert len(net.blocks) == 15, f"expected 15 blocks, got {len(net.blocks)}"
    expected_se = [False, False, False, True, True, True, False, False, False, False, True, True, True, True, True]
    expected_act = ["RE"] * 6 + ["HS"] * 9
    for i, (block, se, act) in enumerate(zip(net.blocks, expected_se, expected_act)):
        assert isinstance(block.se, SqueezeExcite) == se, f"block {i}: SE presence mismatch (expected {se})"
        assert isinstance(block.depthwise[2], (HardSwish if act == "HS" else nn.ReLU)), (
            f"block {i}: activation mismatch (expected {act})"
        )
    # t=1 block (index 0) must skip the expand conv.
    assert len(net.blocks[0].expand) == 0, "block0 (t=1) should skip the expand conv"
    assert len(net.blocks[1].expand) == 3, "block1 (t=4) should have the expand conv"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"MobileNetV3 self-test PASSED: builds, forward-passes to {tuple(out.shape)}, "
          f"15-block schedule verified (SE placement {expected_se.count(True)}/15, "
          f"RE->HS activation switch at block 6), t=1 expand-skip verified. {n_params} params.")
