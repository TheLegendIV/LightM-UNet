from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.nets.ENet import (
    DownsamplingBottleneck,
    InitialBlock,
    RegularBottleneck,
    UpsamplingBottleneck,
)

VALID_EXPERIMENTS = frozenset({
    "baseline",
    "A1", "A2", "A3",
    "B1", "B2",
    "C1", "C2", "C3",
    "D1", "D2", "D3",
    "E1",
    "F1",
})


# ---------------------------------------------------------------------------
# Fusion modules — Groups A / B / C
# ---------------------------------------------------------------------------

class FusionConv1x1(nn.Module):
    """1×1 conv + BN + PReLU: reduces (enc_ch + dec_ch) → dec_ch."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionConv3x3(nn.Module):
    """3×3 conv + BN + PReLU: reduces (enc_ch + dec_ch) → dec_ch with spatial mixing."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionBottleneck(nn.Module):
    """ENet-style bottleneck: (enc_ch + dec_ch) → dec_ch with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4):
        super().__init__()
        internal = max(out_channels // internal_ratio, 1)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=1, bias=False),
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
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.out_act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.residual(x) + self.expand(self.conv(self.reduce(x))))


class AttentionGate(nn.Module):
    """Content-conditional attention gate (additive, Attention U-Net style).

    Projects enc_feat and dec_feat to inter_ch, sums, ReLU → 1ch → Sigmoid.
    Spatial gate multiplies enc_feat — suppresses encoder locations the decoder
    considers irrelevant at the current decoding step.
    """
    def __init__(self, enc_channels: int, dec_channels: int, inter_channels: int):
        super().__init__()
        self.enc_proj = nn.Conv2d(enc_channels, inter_channels, kernel_size=1, bias=False)
        self.dec_proj = nn.Conv2d(dec_channels, inter_channels, kernel_size=1, bias=False)
        self.attn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        gate = self.attn(self.enc_proj(enc_feat) + self.dec_proj(dec_feat))
        return enc_feat * gate


# ---------------------------------------------------------------------------
# Lightweight edge modules — Group D  (pure convolution, no Mamba)
# ---------------------------------------------------------------------------

class LightEdgePrior(nn.Module):
    """Lightweight 1ch edge prior from shallow + deep encoder features.

    Combines x_1x (H/4, low-level spatial detail) with x_3x (H/8, high-level
    context) via 1×1 projections + 3×3 fuse + Sigmoid.
    Params: ~7.7K for (low_ch=64, high_ch=128, edge_ch=16).
    """
    def __init__(self, low_ch: int, high_ch: int, edge_ch: int = 16):
        super().__init__()
        self.low_proj = nn.Conv2d(low_ch, edge_ch, kernel_size=1, bias=False)
        self.high_proj = nn.Conv2d(high_ch, edge_ch, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(edge_ch * 2, edge_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(edge_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        high_up = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.low_proj(low), self.high_proj(high_up)], dim=1))


class LightEdgeGate(nn.Module):
    """Gate features with 1ch edge prior + lightweight 1×1 channel mix.

    Residual gating: x * (1 + edge) preserves the original signal and amplifies
    edge-aligned regions. 1×1 conv lets the network learn which channels benefit
    from the edge modulation.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = F.interpolate(edge, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.mix(x * (1.0 + edge))


# ---------------------------------------------------------------------------
# ENetSkip
# ---------------------------------------------------------------------------

class ENetSkip(nn.Module):
    """ENet with configurable skip connections for ablation study.

    Fixed channel schedule (default, for 512×512 input):
        x_initial : ch_init=16ch,  H/2  — InitialBlock output
        x_1x      : ch_s1=64ch,   H/4  — post stage1 (4 regular blocks)
        x_2x/x_3x : ch_s23=128ch, H/8  — dilated context stages
        x_4x      : ch_s4=64ch,   H/4  — post up4 (MaxUnpool or bilinear)
        x_5x      : ch_s5=16ch,   H/2  — post up5

    MaxUnpool constraint: ch_s4 == ch_s1 and ch_s5 == ch_init (enforced).

    Injection discipline: unpool → [gate] → cat(enc, dec) → fusion → refine.

    Experiments:
        baseline — ENet original, no feature skips
        A1  — raw skip x_1x→4x, Conv1×1 fusion
        A2  — raw skip x_initial→5x, Conv1×1 fusion
        A3  — both raw skips, Conv1×1 fusion
        B1  — both raw skips, Conv3×3 fusion
        B2  — both raw skips, Bottleneck fusion
        C1  — attention-gated skip at 4x, Conv1×1 fusion
        C2  — attention-gated skip at 5x, Conv1×1 fusion
        C3  — attention-gated skips at both points
        D1  — lightweight edge prior gates x_1x at 4x (LightEdgePrior + LightEdgeGate)
        D2  — lightweight edge prior gates at both 4x and 5x
        D3  — edge-gate at 4x + triple concat (x_5x + gate(x_initial) + x_initial) at 5x
        E1  — residual addition at 4x: project x_1x and add to x_4x (no concat, no fusion)
        F1  — bilinear decoder: MaxUnpool → bilinear; no feature skips at all
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, int, int, int, int] = (16, 64, 128, 64, 16),
        experiment: str = "A3",
    ):
        super().__init__()
        if experiment not in VALID_EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{experiment}'. Valid: {sorted(VALID_EXPERIMENTS)}"
            )
        if len(channels) != 5:
            raise ValueError("ENetSkip expects 5 channel values: (initial, stage1, stage23, stage4, stage5).")
        ch_init, ch_s1, ch_s23, ch_s4, ch_s5 = (int(c) for c in channels)
        for ch in (ch_s1, ch_s23, ch_s4, ch_s5):
            if ch % 4 != 0:
                raise ValueError("ENetSkip stage channels must be divisible by 4.")
        if ch_init <= in_channels:
            raise ValueError("initial_channels must exceed in_channels.")
        if ch_s4 != ch_s1 or ch_s5 != ch_init:
            raise ValueError(
                "ENetSkip MaxUnpool constraint: stage4 must equal stage1 and stage5 must equal initial. "
                f"Got: stage4={ch_s4}, stage1={ch_s1}, stage5={ch_s5}, initial={ch_init}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.experiment = experiment

        # --- Standard ENet backbone (identical to ENet.py) ---
        self.initial = InitialBlock(in_channels, ch_init)
        self.down1 = DownsamplingBottleneck(ch_init, ch_s1, dropout_p=0.01)
        self.regular1 = nn.Sequential(
            *[RegularBottleneck(ch_s1, dropout_p=0.01) for _ in range(4)]
        )
        self.down2 = DownsamplingBottleneck(ch_s1, ch_s23, dropout_p=0.1)

        def _make_context_stage(ch: int) -> nn.Sequential:
            return nn.Sequential(
                RegularBottleneck(ch, dropout_p=0.1),
                RegularBottleneck(ch, padding=2, dilation=2, dropout_p=0.1),
                RegularBottleneck(ch, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
                RegularBottleneck(ch, padding=4, dilation=4, dropout_p=0.1),
                RegularBottleneck(ch, dropout_p=0.1),
                RegularBottleneck(ch, padding=8, dilation=8, dropout_p=0.1),
                RegularBottleneck(ch, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
                RegularBottleneck(ch, padding=16, dilation=16, dropout_p=0.1),
            )

        self.stage2 = _make_context_stage(ch_s23)
        self.stage3 = _make_context_stage(ch_s23)

        # up4: ch_s23→ch_s4; MaxUnpool constraint: ch_s4==ch_s1 (idx2 has ch_s1 channels) ✓
        # up5: ch_s4→ch_s5;  MaxUnpool constraint: ch_s5==ch_init (idx1 has ch_init channels) ✓
        self.up4 = UpsamplingBottleneck(ch_s23, ch_s4)
        self.regular4 = nn.Sequential(
            RegularBottleneck(ch_s4, dropout_p=0.1, relu=True),
            RegularBottleneck(ch_s4, dropout_p=0.1, relu=True),
        )
        self.up5 = UpsamplingBottleneck(ch_s4, ch_s5)
        self.regular5 = RegularBottleneck(ch_s5, dropout_p=0.1, relu=True)
        self.final = nn.ConvTranspose2d(ch_s5, out_channels, kernel_size=2, stride=2)

        # --- Skip / structural modules (experiment-dependent) ---
        self._setup_skip_modules(ch_init, ch_s1, ch_s23, ch_s4, ch_s5, experiment)

    # ------------------------------------------------------------------
    def _setup_skip_modules(
        self,
        ch_init: int, ch_s1: int, ch_s23: int, ch_s4: int, ch_s5: int,
        experiment: str,
    ) -> None:
        # Control flags (Python bools, not tracked by PyTorch)
        self.skip_1x: bool = False        # save x_1x and inject at 4x
        self.skip_initial: bool = False   # save x_initial and inject at 5x
        self.residual_4x: bool = False    # E1: additive residual, no cat at 4x
        self.d3_triple_5x: bool = False   # D3: triple cat at 5x
        self.bilinear_up: bool = False    # F1: bypass MaxUnpool indices

        # Optional nn.Module slots (registered when set to a module)
        self.fuse_4x: nn.Module | None = None
        self.fuse_5x: nn.Module | None = None
        self.attn_4x: nn.Module | None = None
        self.attn_5x: nn.Module | None = None
        self.lep: nn.Module | None = None       # LightEdgePrior (D group)
        self.leg_4x: nn.Module | None = None    # LightEdgeGate at 4x (D group)
        self.leg_5x: nn.Module | None = None    # LightEdgeGate at 5x (D2/D3)
        self.proj_enc_4x: nn.Module | None = None  # residual projection (E1)

        def make_fuse_4x(kind: str) -> nn.Module:
            in_ch = ch_s1 + ch_s4  # 64+64=128
            if kind == "conv1x1":    return FusionConv1x1(in_ch, ch_s4)
            if kind == "conv3x3":    return FusionConv3x3(in_ch, ch_s4)
            if kind == "bottleneck": return FusionBottleneck(in_ch, ch_s4)

        def make_fuse_5x(kind: str, in_ch: int | None = None) -> nn.Module:
            if in_ch is None:
                in_ch = ch_init + ch_s5  # 16+16=32
            if kind == "conv1x1":    return FusionConv1x1(in_ch, ch_s5)
            if kind == "conv3x3":    return FusionConv3x3(in_ch, ch_s5)
            if kind == "bottleneck": return FusionBottleneck(in_ch, ch_s5)

        # --- Group A: raw skip, Conv1×1 fusion ---
        if experiment == "A1":
            self.skip_1x = True
            self.fuse_4x = make_fuse_4x("conv1x1")

        elif experiment == "A2":
            self.skip_initial = True
            self.fuse_5x = make_fuse_5x("conv1x1")

        elif experiment == "A3":
            self.skip_1x = True
            self.skip_initial = True
            self.fuse_4x = make_fuse_4x("conv1x1")
            self.fuse_5x = make_fuse_5x("conv1x1")

        # --- Group B: raw skip, fusion depth ---
        elif experiment == "B1":
            self.skip_1x = True
            self.skip_initial = True
            self.fuse_4x = make_fuse_4x("conv3x3")
            self.fuse_5x = make_fuse_5x("conv3x3")

        elif experiment == "B2":
            self.skip_1x = True
            self.skip_initial = True
            self.fuse_4x = make_fuse_4x("bottleneck")
            self.fuse_5x = make_fuse_5x("bottleneck")

        # --- Group C: attention-gated skip ---
        elif experiment == "C1":
            self.skip_1x = True
            self.attn_4x = AttentionGate(ch_s1, ch_s4, inter_channels=32)
            self.fuse_4x = make_fuse_4x("conv1x1")

        elif experiment == "C2":
            self.skip_initial = True
            self.attn_5x = AttentionGate(ch_init, ch_s5, inter_channels=8)
            self.fuse_5x = make_fuse_5x("conv1x1")

        elif experiment == "C3":
            self.skip_1x = True
            self.skip_initial = True
            self.attn_4x = AttentionGate(ch_s1, ch_s4, inter_channels=32)
            self.attn_5x = AttentionGate(ch_init, ch_s5, inter_channels=8)
            self.fuse_4x = make_fuse_4x("conv1x1")
            self.fuse_5x = make_fuse_5x("conv1x1")

        # --- Group D: lightweight edge-guided skip (pure convolution, no Mamba) ---
        elif experiment in ("D1", "D2", "D3"):
            self.skip_1x = True
            # LightEdgePrior: low=x_1x (ch_s1), deep=x_3x (ch_s23) → 1ch edge map
            self.lep = LightEdgePrior(ch_s1, ch_s23, edge_ch=16)
            self.leg_4x = LightEdgeGate(ch_s1)
            self.fuse_4x = make_fuse_4x("conv1x1")

            if experiment in ("D2", "D3"):
                self.skip_initial = True
                self.leg_5x = LightEdgeGate(ch_init)
                if experiment == "D3":
                    # Triple cat: x_5x + gate(x_initial) + raw x_initial → 3×ch_init
                    self.d3_triple_5x = True
                    self.fuse_5x = make_fuse_5x("conv1x1", in_ch=ch_s5 + ch_init + ch_init)
                else:
                    self.fuse_5x = make_fuse_5x("conv1x1")

        # --- E1: residual addition at 4x only (no concat, no dedicated fusion block) ---
        elif experiment == "E1":
            self.skip_1x = True
            self.residual_4x = True
            self.proj_enc_4x = nn.Sequential(
                nn.Conv2d(ch_s1, ch_s4, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch_s4),
                nn.PReLU(ch_s4),
            )

        # --- F1: bilinear decoder — MaxUnpool → bilinear; no feature skips ---
        elif experiment == "F1":
            self.bilinear_up = True

        # baseline: no modifications

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # === Encoder ===
        x = self.initial(x)
        x_initial = x                         # ch_init ch, H/2

        x, idx1, sz1 = self.down1(x)         # ch_init→ch_s1
        x = self.regular1(x)
        x_1x = x                              # ch_s1 ch, H/4

        x, idx2, sz2 = self.down2(x)         # ch_s1→ch_s23
        x = self.stage2(x)
        x = self.stage3(x)
        x_3x = x                              # ch_s23 ch, H/8

        # Lightweight edge prior — computed once for Group D experiments
        fe = self.lep(x_1x, x_3x) if self.lep is not None else None

        # === Decoder ===
        # F1: pass None indices → UpsamplingBottleneck falls back to bilinear
        up4_idx = None if self.bilinear_up else idx2
        up5_idx = None if self.bilinear_up else idx1

        x = self.up4(x, sz2, up4_idx)        # ch_s23→ch_s4, H/8→H/4
        x_4x = x                              # post-unpool / post-bilinear, H/4

        # Injection at 4x
        if self.skip_1x:
            if self.residual_4x:
                # E1: project enc channels and add — no concat, no fusion block
                x = x_4x + self.proj_enc_4x(x_1x)
            else:
                enc = x_1x
                if self.attn_4x is not None:
                    enc = self.attn_4x(enc, x_4x)
                elif self.leg_4x is not None:
                    enc = self.leg_4x(enc, fe)
                x = self.fuse_4x(torch.cat([enc, x_4x], dim=1))

        x = self.regular4(x)

        x = self.up5(x, sz1, up5_idx)        # ch_s4→ch_s5, H/4→H/2
        x_5x = x                              # post-unpool / post-bilinear, H/2

        # Injection at 5x
        if self.skip_initial:
            enc = x_initial
            if self.attn_5x is not None:
                enc = self.attn_5x(enc, x_5x)
            elif self.leg_5x is not None:
                enc = self.leg_5x(enc, fe)

            if self.d3_triple_5x:
                # D3: x_5x + gate(x_initial) + raw x_initial → 3×ch_init → fuse → ch_init
                x = self.fuse_5x(torch.cat([x_5x, enc, x_initial], dim=1))
            else:
                x = self.fuse_5x(torch.cat([enc, x_5x], dim=1))

        x = self.regular5(x)
        x = self.final(x)                     # ch_s5→out_ch, H/2→H
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
