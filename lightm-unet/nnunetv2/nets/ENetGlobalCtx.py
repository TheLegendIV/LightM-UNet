"""ENet Global Context Search — G1 through G11.

Each experiment adds or replaces components with a global context mechanism
(Mamba SSM, self-attention, linear attention) relative to the ENet baseline.
Fixed channel schedule for all experiments: (20, 72, 144, 72, 20).

  G1  — +1 PVMamba at bridge (after stage 3, cheapest global context probe)
  G2  — stage 3 replaced by 4 PVMamba blocks (forward scan)
  G3  — stage 3 replaced by 4 PVMamba blocks, alternating scan directions
  G4  — stages 2 and 3 both replaced by 4 PVMamba blocks each
  G5  — +1 PVMamba after down2 + +1 PVMamba after stage 3 (transition probes)
  G6  — parallel PVMamba branch at stages 2+3, concat, ECA to select
  G7  — +1 PVMamba at decoder start (after up4, H/4 = 16 K tokens)
  G8  — +1 full multi-head self-attention at bridge (O(N²), N=4096)
  G9  — +1 windowed self-attention at bridge, 8×8 windows (O(N·w²))
  G10 — cross-depth Mamba: stages 2+3 features concatenated, 8192 tokens
  G11 — +1 linear attention at stage 1 (H/4, 16 384 tokens, O(N))
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from nnunetv2.nets.ENet import (
    DownsamplingBottleneck,
    ENet,
    InitialBlock,
    RegularBottleneck,
    UpsamplingBottleneck,
)

CHANNELS = (20, 72, 144, 72, 20)
VALID_EXPERIMENTS = frozenset(f"G{i}" for i in range(1, 12))


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def _ctx_stage(ch: int) -> nn.Sequential:
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


def _s1_stage(ch: int) -> nn.Sequential:
    return nn.Sequential(*[RegularBottleneck(ch, dropout_p=0.01) for _ in range(4)])


def _dec_ref(ch: int) -> nn.Sequential:
    return nn.Sequential(RegularBottleneck(ch, relu=True), RegularBottleneck(ch, relu=True))


# ---------------------------------------------------------------------------
# Global context blocks
# ---------------------------------------------------------------------------

class PVMambaBlock(nn.Module):
    """Plain Vision Mamba: (B,C,H,W) → flatten → Mamba SSM → reshape + residual.

    scan_dir: 0=row-forward  1=row-backward  2=col-forward  3=col-backward
    """

    def __init__(self, dim: int, scan_dir: int = 0,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.scan_dir = scan_dir % 4
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def _to_seq(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        B, C, H, W = x.shape
        sd = self.scan_dir
        if sd == 0:
            s = x.flatten(2).transpose(1, 2)
        elif sd == 1:
            s = x.flatten(2).transpose(1, 2).flip(1)
        elif sd == 2:
            s = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)
        else:
            s = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2).flip(1)
        return s, (B, C, H, W)

    def _from_seq(self, s: torch.Tensor, meta: tuple) -> torch.Tensor:
        B, C, H, W = meta
        sd = self.scan_dir
        if sd == 0:
            return s.transpose(1, 2).reshape(B, C, H, W)
        elif sd == 1:
            return s.flip(1).transpose(1, 2).reshape(B, C, H, W)
        elif sd == 2:
            return s.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        else:
            return s.flip(1).transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, meta = self._to_seq(x)
        return self._from_seq(s + self.mamba(self.norm(s)), meta)


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al. 2020)."""

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x).squeeze(-1).transpose(-1, -2)          # B, 1, C
        y = self.act(self.conv(y)).transpose(-1, -2).unsqueeze(-1)  # B, C, 1, 1
        return x * y


class BridgeSelfAttention(nn.Module):
    """Standard multi-head self-attention at bridge (O(N²), N = H×W)."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = x.flatten(2).transpose(1, 2)       # B, N, C
        sn = self.norm(s)
        out, _ = self.attn(sn, sn, sn)
        return (s + out).transpose(1, 2).reshape(B, C, H, W)


class WindowedSelfAttention(nn.Module):
    """Self-attention within non-overlapping spatial windows (O(N·w²))."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8):
        super().__init__()
        self.ws = window_size
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.ws
        nh, nw = H // ws, W // ws
        xw = x.reshape(B, C, nh, ws, nw, ws).permute(0, 2, 4, 3, 5, 1)
        xw = xw.reshape(B * nh * nw, ws * ws, C)
        xn = self.norm(xw)
        out, _ = self.attn(xn, xn, xn)
        xw = xw + out
        xw = xw.reshape(B, nh, nw, ws, ws, C).permute(0, 5, 1, 3, 2, 4)
        return xw.reshape(B, C, H, W)


class LinearAttentionBlock(nn.Module):
    """Linear attention via ELU+1 feature map — O(N) memory.

    Safe for high-resolution stages (e.g. H/4 = 128×128 = 16 384 tokens).
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _phi(t: torch.Tensor) -> torch.Tensor:
        return F.elu(t) + 1.0   # non-negative feature map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        s = x.flatten(2).transpose(1, 2)   # B, N, C
        sn = self.norm(s)
        qkv = self.qkv(sn).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each: B, heads, N, D
        q, k = self._phi(q), self._phi(k)
        kv = torch.einsum("bhnd,bhnv->bhdv", k, v)               # B, heads, D, D
        denom = torch.einsum("bhnd,bhd->bhn", q, k.sum(2))        # B, heads, N
        out = torch.einsum("bhnd,bhdv->bhnv", q, kv)              # B, heads, N, D
        out = out / denom.clamp(min=1e-6).unsqueeze(-1)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return (s + self.proj(out)).transpose(1, 2).reshape(B, C, H, W)


class CrossDepthMambaBlock(nn.Module):
    """Jointly process stage-2 and stage-3 features as a single 8192-token sequence."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, feat2: torch.Tensor, feat3: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat3.shape
        s2 = feat2.flatten(2).transpose(1, 2)   # B, N, C
        s3 = feat3.flatten(2).transpose(1, 2)   # B, N, C
        combined = torch.cat([s2, s3], dim=1)   # B, 2N, C
        out = self.mamba(self.norm(combined))    # B, 2N, C
        # Stage-3 half with residual — stage-2 tokens provided context via Mamba causality
        out3 = out[:, H * W:, :] + s3
        return out3.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# G1 — single PVMamba at bridge
# ---------------------------------------------------------------------------

class G1(nn.Module):
    """Minimal global context probe: one PVMamba block inserted at the bridge."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.bridge = PVMambaBlock(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bridge(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G2 — stage 3 replaced by 4 PVMamba (forward scan)
# ---------------------------------------------------------------------------

class G2(nn.Module):
    """Stage 3 dilated conv blocks replaced by 4 PVMamba blocks (single direction)."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = nn.ModuleList([PVMambaBlock(ch23, scan_dir=0) for _ in range(4)])
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        for blk in self.stage3:
            x = blk(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G3 — stage 3 replaced by 4 PVMamba, alternating 4 scan directions
# ---------------------------------------------------------------------------

class G3(nn.Module):
    """Stage 3 replaced by 4 directional PVMamba blocks (0→1→2→3 scan dirs)."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = nn.ModuleList([PVMambaBlock(ch23, scan_dir=i) for i in range(4)])
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        for blk in self.stage3:
            x = blk(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G4 — stages 2 and 3 both replaced by 4 PVMamba each
# ---------------------------------------------------------------------------

class G4(nn.Module):
    """Both stage 2 and stage 3 replaced by 4 PVMamba blocks each."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = nn.ModuleList([PVMambaBlock(ch23, scan_dir=0) for _ in range(4)])
        self.stage3 = nn.ModuleList([PVMambaBlock(ch23, scan_dir=0) for _ in range(4)])
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G5 — PVMamba added at transition points (after down2 and after stage3)
# ---------------------------------------------------------------------------

class G5(nn.Module):
    """Global context injected at stage boundaries rather than within stages."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.post_down2 = PVMambaBlock(ch23)   # transition: entry into H/8
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.post_stage3 = PVMambaBlock(ch23)  # transition: bridge before decoder
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.post_down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.post_stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G6 — parallel PVMamba branch, concat, ECA fusion
# ---------------------------------------------------------------------------

def _parallel_fuse(ch: int) -> nn.Sequential:
    return nn.Sequential(
        ECA(ch * 2),
        nn.Conv2d(ch * 2, ch, 1, bias=False),
        nn.BatchNorm2d(ch),
        nn.PReLU(ch),
    )


class G6(nn.Module):
    """Conv and Mamba branches run in parallel at stages 2+3; ECA selects."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.mamba2 = PVMambaBlock(ch23)
        self.fuse2 = _parallel_fuse(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.mamba3 = PVMambaBlock(ch23)
        self.fuse3 = _parallel_fuse(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.fuse2(torch.cat([self.stage2(x), self.mamba2(x)], dim=1))
        x = self.fuse3(torch.cat([self.stage3(x), self.mamba3(x)], dim=1))
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G7 — PVMamba at start of decoder (after up4, H/4 = 16 K tokens, dim=ch4)
# ---------------------------------------------------------------------------

class G7(nn.Module):
    """Global context during decoder reconstruction at H/4 resolution."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.dec_mamba = PVMambaBlock(ch4)      # dim=ch4 at H/4
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.up4(x, s2, idx2)
        x = self.dec_mamba(x)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G8 — full self-attention at bridge (O(N²))
# ---------------------------------------------------------------------------

class G8(nn.Module):
    """Standard multi-head self-attention at bridge. Pairwise content-conditional."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.bridge_attn = BridgeSelfAttention(ch23, num_heads=8)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bridge_attn(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G9 — windowed self-attention at bridge
# ---------------------------------------------------------------------------

class G9(nn.Module):
    """Windowed self-attention (8×8 patches) at bridge — local pairwise attention."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.bridge_attn = WindowedSelfAttention(ch23, window_size=8, num_heads=8)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bridge_attn(x)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G10 — cross-depth Mamba (stages 2+3 together, 8192 tokens)
# ---------------------------------------------------------------------------

class G10(nn.Module):
    """Mamba processes stage-2 and stage-3 features jointly as 8192-token sequence."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.cross_depth = CrossDepthMambaBlock(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x, idx2, s2 = self.down2(x)
        feat2 = self.stage2(x)
        feat3 = self.stage3(feat2)
        x = self.cross_depth(feat2, feat3)
        x = self.up4(x, s2, idx2)
        x = self.regular4(x)
        x = self.up5(x, s1, idx1)
        x = self.regular5(x)
        x = self.final(x)
        if x.shape[2:] != sz:
            x = F.interpolate(x, size=sz, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# G11 — linear attention at stage 1 (H/4 = 128×128 = 16 384 tokens)
# ---------------------------------------------------------------------------

class G11(nn.Module):
    """Linear attention at H/4 — global context before spatial information is pooled."""

    def __init__(self, in_channels=1, out_channels=4, channels=CHANNELS):
        super().__init__()
        ch0, ch1, ch23, ch4, ch5 = channels
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = _s1_stage(ch1)
        self.lin_attn = LinearAttentionBlock(ch1, num_heads=8)   # dim=ch1=72
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)
        self.stage2 = _ctx_stage(ch23)
        self.stage3 = _ctx_stage(ch23)
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, 2, stride=2)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.initial(x)
        x, idx1, s1 = self.down1(x)
        x = self.stage1(x)
        x = self.lin_attn(x)      # H/4 linear attention
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
    "G1": G1, "G2": G2, "G3": G3, "G4": G4,
    "G5": G5, "G6": G6, "G7": G7, "G8": G8,
    "G9": G9, "G10": G10, "G11": G11,
}


def build_gctx_model(
    experiment: str,
    in_channels: int = 1,
    out_channels: int = 4,
    channels: tuple = CHANNELS,
) -> nn.Module:
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"Unknown GCX experiment '{experiment}'. Valid: {sorted(VALID_EXPERIMENTS)}"
        )
    return _REGISTRY[experiment](
        in_channels=in_channels, out_channels=out_channels, channels=channels
    )
