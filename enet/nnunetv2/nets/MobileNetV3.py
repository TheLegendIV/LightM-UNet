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
    computed from the block's own INPUT channels (pre-expansion), per the
    source's own squeeze_factor=4 applied at `ops.squeeze_excite` call
    time (before expansion happens in the real graph) -- see this file's
    own module docstring for why that's the detail most implementations
    get backwards."""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        squeeze_channels = _make_divisible(in_channels / 4, 8)
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
        expand_ratio: float, use_se: bool, act: str,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"InvertedResidualV3 stride must be 1 or 2, got {stride}.")
        hidden_dim = _make_divisible(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        padding = (kernel_size - 1) // 2

        expand: list[nn.Module] = []
        if expand_ratio != 1:
            expand = [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim), _activation(act),
            ]
        self.expand = nn.Sequential(*expand)

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), _activation(act),
        )
        self.se = SqueezeExcite(in_channels, hidden_dim) if use_se else nn.Identity()
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
# 0-indexed rows whose OWN output sits at each of the 4 non-stem downsample
# depths' deepest point (i.e. the last row before the NEXT stride-2 row) --
# used to pick the decoder's own mirrored channel schedule, same convention
# MobileNetV2.py's own decoder construction uses.
_DEPTH_MARKER_ROWS = (0, 2, 5, 11)  # depths /2, /4, /8, /16 respectively (row 14 / head2 covers /32)


class MobileNetV3(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 20,
        width_mult: float = 1.0,
        stem_channels: int = 16,
        head1_channels: int = 960,
        head2_channels: int = 1280,
    ):
        super().__init__()
        stem_c = _make_divisible(stem_channels * width_mult)

        # -- Encoder (real architecture, /32 total downsample) --------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_c), HardSwish(),
        )

        self.blocks = nn.ModuleList()
        in_c = stem_c
        depth_marker_channels: dict[int, int] = {}
        for i, (t, c, k, s, se, act) in enumerate(_LARGE_SETTING):
            out_c = _make_divisible(c * width_mult)
            self.blocks.append(InvertedResidualV3(in_c, out_c, k, s, t, se, act))
            in_c = out_c
            if i in _DEPTH_MARKER_ROWS:
                depth_marker_channels[i] = out_c

        head1_c = _make_divisible(head1_channels * width_mult)
        head2_c = _make_divisible(head2_channels * max(1.0, width_mult))
        self.head = nn.Sequential(
            nn.Conv2d(in_c, head1_c, kernel_size=1, bias=False), nn.BatchNorm2d(head1_c), HardSwish(),
            nn.Conv2d(head1_c, head2_c, kernel_size=1, bias=True), HardSwish(),  # source: normalizer_fn=None here
        )

        # -- Decoder (this repo's own addition -- see module docstring) -----
        d0, d1, d2, d3 = (depth_marker_channels[i] for i in (11, 5, 2, 0))
        self.up1 = UpsampleBlockHS(head2_c, d0)
        self.up2 = UpsampleBlockHS(d0, d1)
        self.up3 = UpsampleBlockHS(d1, d2)
        self.up4 = UpsampleBlockHS(d2, d3)
        self.up5 = UpsampleBlockHS(d3, stem_c)
        self.final = nn.Conv2d(stem_c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
