"""ENet Upscale Architectural Search — UAS0 through UAS8.

Each UASn isolates one structural modification vs ENetOriginal (UAS0).
Fixed channel schedule for all experiments: (20, 72, 144, 72, 20).

  UAS0 — baseline (ENetOriginal)
  UAS1 — extra dilated stage at H/16 (3rd downsampling)
  UAS2 — +2 bottlenecks in stages 2+3 (8→10 each)
  UAS3 — stage 2 flat at H/4, second downsample delayed to stage-3 entry
  UAS4 — UAS3 + retuned dilation rates 4/8/16/32
  UAS5 — double full context cycle in stages 2+3 (8→16 blocks each)
  UAS6 — learnable 2×2 strided conv replaces MaxPool in both downsampling bottlenecks
  UAS7 — stages 2+3 with internal_ratio=2 (bottleneck projects to ch/2 not ch/4)
  UAS8 — stage 1 uses asymmetric 5×1/1×5 convolutions (anisotropic bias at H/4)
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.nets.ENet import (
    DownsamplingBottleneck,
    ENet,
    InitialBlock,
    RegularBottleneck,
    UpsamplingBottleneck,
)

CHANNELS = (20, 72, 144, 72, 20)
VALID_EXPERIMENTS = frozenset(
    {"UAS0", "UAS1", "UAS2", "UAS3", "UAS4", "UAS5", "UAS6", "UAS7", "UAS8"}
)


# ---------------------------------------------------------------------------
# Context-stage builders
# ---------------------------------------------------------------------------

def _ctx(ch: int, dils: tuple = (2, 4, 8, 16), internal_ratio: int = 4) -> nn.Sequential:
    """Standard 8-block dilated context stage.

    dils: four dilation values for the non-regular, non-asym blocks.
    internal_ratio: bottleneck projection ratio (default 4 → ch/4 internal channels).
    """
    d1, d2, d3, d4 = dils
    r = internal_ratio
    return nn.Sequential(
        RegularBottleneck(ch, internal_ratio=r, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, padding=d1, dilation=d1, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, padding=d2, dilation=d2, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, padding=d3, dilation=d3, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
        RegularBottleneck(ch, internal_ratio=r, padding=d4, dilation=d4, dropout_p=0.1),
    )


def _ext_ctx(ch: int) -> nn.Sequential:
    """10-block context stage: standard 8 + 2 extra (dil4, dil8)."""
    return nn.Sequential(
        RegularBottleneck(ch, dropout_p=0.1),
        RegularBottleneck(ch, padding=2, dilation=2, dropout_p=0.1),
        RegularBottleneck(ch, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
        RegularBottleneck(ch, padding=4, dilation=4, dropout_p=0.1),
        RegularBottleneck(ch, dropout_p=0.1),
        RegularBottleneck(ch, padding=8, dilation=8, dropout_p=0.1),
        RegularBottleneck(ch, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
        RegularBottleneck(ch, padding=16, dilation=16, dropout_p=0.1),
        RegularBottleneck(ch, padding=4, dilation=4, dropout_p=0.1),
        RegularBottleneck(ch, padding=8, dilation=8, dropout_p=0.1),
    )


def _double_ctx(ch: int) -> nn.Sequential:
    """16-block context stage: two consecutive complete 8-block dilation cycles."""
    def _cycle() -> list:
        return [
            RegularBottleneck(ch, dropout_p=0.1),
            RegularBottleneck(ch, padding=2, dilation=2, dropout_p=0.1),
            RegularBottleneck(ch, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
            RegularBottleneck(ch, padding=4, dilation=4, dropout_p=0.1),
            RegularBottleneck(ch, dropout_p=0.1),
            RegularBottleneck(ch, padding=8, dilation=8, dropout_p=0.1),
            RegularBottleneck(ch, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
            RegularBottleneck(ch, padding=16, dilation=16, dropout_p=0.1),
        ]
    return nn.Sequential(*_cycle(), *_cycle())


def _dec_ref(ch: int) -> nn.Sequential:
    return nn.Sequential(RegularBottleneck(ch, relu=True), RegularBottleneck(ch, relu=True))


# ---------------------------------------------------------------------------
# Learnable downsampling block (UAS6)
# ---------------------------------------------------------------------------

class LearnableDownsamplingBottleneck(nn.Module):
    """DownsamplingBottleneck with a learned 2×2 strided conv replacing MaxPool.

    The MaxPool-based main branch is fixed and winner-take-all; this variant
    makes spatial aggregation trainable. No MaxUnpool indices are produced —
    the corresponding decoder must use bilinear upsampling (indices=None).
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.01):
        super().__init__()
        internal = out_channels // 4
        # Learned stride-2 conv: same spatial reduction as MaxPool but trainable
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal),
            nn.PReLU(internal),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(internal, internal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal),
            nn.PReLU(internal),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.out_act = nn.PReLU(out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        input_size = x.size()
        main = self.main_conv(x)
        if main.shape[1] < self.out_channels:
            pad = torch.zeros(
                main.shape[0], self.out_channels - main.shape[1],
                main.shape[2], main.shape[3],
                dtype=main.dtype, device=main.device,
            )
            main = torch.cat([main, pad], dim=1)
        ext = self.reduce(x)
        ext = self.conv(ext)
        ext = self.dropout(self.expand(ext))
        return self.out_act(main + ext), None, input_size


# ---------------------------------------------------------------------------
# UAS0 — baseline (alias)
# ---------------------------------------------------------------------------

UAS0 = ENet


# ---------------------------------------------------------------------------
# UAS1 — extra dilated stage at H/16
# ---------------------------------------------------------------------------

class UAS1(nn.Module):
    """Add a 3rd downsampling to H/16 + 8-block context stage + decoder stage.

    Hypothesis: deeper hierarchy captures vessel branching topology and
    long-range class identity that the 2-stage ENet cannot represent.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx(ch23)
        self.stage3 = _ctx(ch23)
        # H/16 stage — repeat stage-3 dilation pattern
        self.down3 = DownsamplingBottleneck(ch23, ch23, dropout_p=0.1)
        self.stage4 = _ctx(ch23)
        # MaxUnpool: up_deep out must == ch23 (idx3 from down3 which received ch23 input)
        self.up_deep = UpsamplingBottleneck(ch23, ch23)
        self.ref_deep = _dec_ref(ch23)
        # Original 2-level decoder unchanged
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x, idx3, s3 = self.down3(x)
        x = self.stage4(x)
        x = self.up_deep(x, s3, idx3)
        x = self.ref_deep(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS2 — extra bottlenecks in stages 2+3
# ---------------------------------------------------------------------------

class UAS2(nn.Module):
    """Stages 2 and 3 extended from 8 to 10 blocks each (+dil4, +dil8).

    Hypothesis: more depth at fixed H/8 resolution improves vessel feature
    abstraction without resolution or receptive-field changes.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ext_ctx(ch23)
        self.stage3 = _ext_ctx(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS3 — delayed second downsample (stage 2 flat at H/4)
# ---------------------------------------------------------------------------

class UAS3(nn.Module):
    """Stage 2 runs as a flat dilated context stage at H/4; stride is delayed to stage-3 entry.

    Hypothesis: preserving higher resolution through the semantic aggregation
    stage improves survival of thin vessel structures before spatial information
    is discarded by downsampling.

    MaxUnpool: idx2 originates from down2(ch1 input) → up4 out must == ch1 == ch4.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        # Flat dilated context at H/4 using stage-1 channels
        self.stage2_flat = _ctx(ch1)
        # Stride delayed: now operates on ch1 (H/4 → H/8)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage3 = _ctx(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x = self.stage2_flat(x)       # H/4, no stride
        x, idx2, s2 = self.down2(x)  # stride here
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS4 — UAS3 + retuned dilation rates 4/8/16/32
# ---------------------------------------------------------------------------

class UAS4(nn.Module):
    """UAS3 with dilation rates doubled to 4/8/16/32 in both flat and deep stages.

    Hypothesis: the flat H/4 stage has twice the spatial resolution of ENet's H/8
    stage, so proportional receptive field coverage requires doubling the dilation
    rates. UAS4 tests whether compensating for this recovers context lost vs UAS3.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.stage2_flat = _ctx(ch1, dils=(4, 8, 16, 32))
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage3 = _ctx(ch23, dils=(4, 8, 16, 32))
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x = self.stage2_flat(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS5 — double full context cycle in stages 2+3
# ---------------------------------------------------------------------------

class UAS5(nn.Module):
    """Stages 2 and 3 each run two complete 8-block dilation cycles (8→16 blocks).

    Hypothesis: a second full pass through the dilation/asymmetric/regular
    sequence provides a second round of context integration at different
    receptive-field scales, potentially capturing vessel structures missed
    in the first cycle.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _double_ctx(ch23)
        self.stage3 = _double_ctx(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS6 — learnable strided-conv downsampling
# ---------------------------------------------------------------------------

class UAS6(nn.Module):
    """Replaces MaxPool2d in both downsampling bottlenecks with a 2×2 strided conv.

    ENet's MaxPool discards all non-maximal activations — a non-learnable,
    winner-take-all operation that may suppress thin vessel pixels at boundaries.
    A learned strided conv can be trained to aggregate spatially rather than select.
    Decoder uses bilinear upsampling (no indices available from strided-conv path).

    Hypothesis: learnable downsampling preserves vessel-relevant features that
    MaxPool discards, improving thin-structure detection.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = LearnableDownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.down2 = LearnableDownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx(ch23)
        self.stage3 = _ctx(ch23)
        # Bilinear decoder (no MaxUnpool indices from learnable downsampling)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, _, s1 = self.down1(x)   # indices=None
        x = self.stage1(x)
        x, _, s2 = self.down2(x)   # indices=None
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, None)  # bilinear
        x = self.regular4(x)
        x = self.up5(x, s1, None)  # bilinear
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS7 — wider bottleneck compression in stages 2+3
# ---------------------------------------------------------------------------

class UAS7(nn.Module):
    """Stages 2+3 use internal_ratio=2 — bottleneck projects to ch/2 (72ch) not ch/4 (36ch).

    ENet's default ratio of 4 compresses 144 channels to 36 before the spatial
    convolution — aggressive compression chosen for road-scene efficiency. For
    thin-structure segmentation, this may discard discriminative features.
    Doubling the internal channels widens the spatial conv's representational
    capacity without changing architecture depth or resolution.

    Hypothesis: ENet's ch/4 compression is too aggressive for thin vessel
    structures; relaxing to ch/2 improves discrimination at the deepest stage.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx(ch23, internal_ratio=2)
        self.stage3 = _ctx(ch23, internal_ratio=2)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# UAS8 — asymmetric convolutions extended to stage 1
# ---------------------------------------------------------------------------

class UAS8(nn.Module):
    """Stage 1 uses asymmetric 5×1/1×5 convolutions instead of plain 3×3.

    ENet applies asymmetric convolutions only in the dilated context stages
    (2+3) at H/8. Stage 1's four regular bottlenecks at H/4 use standard 3×3
    convolutions. Vessels are elongated structures; applying anisotropic kernels
    at H/4 — where vessel structures are still clearly spatially extended —
    may capture orientation-sensitive features before spatial information is
    pooled away by the second downsampling.

    Hypothesis: asymmetric kernels at H/4 provide orientation-sensitive features
    for elongated structures that isotropic 3×3 convolutions cannot.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, channels: tuple = CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        # Stage 1: asymmetric bottlenecks at H/4
        self.stage1 = nn.Sequential(*[
            RegularBottleneck(ch1, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.01)
            for _ in range(4)
        ])
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx(ch23)
        self.stage3 = _ctx(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "UAS0": UAS0,
    "UAS1": UAS1,
    "UAS2": UAS2,
    "UAS3": UAS3,
    "UAS4": UAS4,
    "UAS5": UAS5,
    "UAS6": UAS6,
    "UAS7": UAS7,
    "UAS8": UAS8,
}


def build_uas_model(
    experiment: str,
    in_channels: int = 1,
    out_channels: int = 4,
    channels: tuple = CHANNELS,
) -> nn.Module:
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"Unknown UAS experiment '{experiment}'. Valid: {sorted(VALID_EXPERIMENTS)}"
        )
    return _REGISTRY[experiment](
        in_channels=in_channels, out_channels=out_channels, channels=channels
    )
