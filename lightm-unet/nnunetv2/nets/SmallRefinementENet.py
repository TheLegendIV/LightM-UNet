from __future__ import annotations

import torch
from torch import nn

from nnunetv2.nets.ENet import DownsamplingBottleneck, RegularBottleneck, UpsamplingBottleneck

"""
Two-channel (raw image + first-pass predicted mask) ENet-lite for
Dataset507_ARCADE_refinement, per analysis/507_refinement_net_plan.md.

    image  (1ch) -> stem_img  (2x 3x3 conv, 8ch)  -\
                                                      >-- concat (16ch) -> 1x1 fusion conv -> Cw ch, full res
    pred   (1ch) -> stem_pred (2x 3x3 conv, 8ch)  -/
      -> DownsamplingBottleneck (Cw -> 2Cw)              [one pooling stage, H/2]
      -> 3x RegularBottleneck (dilation 1, 2, 4)         [cheap context at half-res]
      -> UpsamplingBottleneck (2Cw -> Cw)                [restore full res]
      -> concat with pre-downsample fused features (2Cw) -> 1x1 conv -> Cw
      -> RegularBottleneck (dilation 1)
      -> RegularBottleneck (kernel=5, asymmetric)         [sharpen thin structure post-roundtrip]
      -> 1x1 conv head -> 1 logit channel

Separate stems let each input develop channel-appropriate low-level features
before anything mixes (image intensity/gradients vs. a binary trust/error
signal are not the same kind of thing) -- the actual fusion decision is
concentrated in the two explicit 1x1 "fusion conv" projections (post-stem,
and post-skip-concat) rather than diffused across a naive single shared
first layer. Exactly one down/up round-trip, not the full ENet depth: 96x96
already only has one halving of headroom before a 1-2px vessel risks
vanishing between pixels, so this deliberately stops after one instead of
"downsizing freely."

Returns raw logits (no sigmoid) -- same convention as SmallENet; apply
torch.sigmoid(...) externally, DC_and_BCE_and_clDice_loss needs raw logits.

Unlike SmallENet, this is NOT resolution-agnostic: UpsamplingBottleneck's
ConvTranspose2d branch doesn't receive an explicit target size the way its
max-unpool branch does (see ENet.py's own error message), so H and W must
both be even for the pool -> unpool round-trip to agree on shape. Fine in
practice -- nnU-Net resamples Dataset507_ARCADE_refinement's 85x85 raw
patches to 96x96 during preprocessing (confirmed for this whole ARCADE
patch-dataset family) -- but will raise a clear RuntimeError from
UpsamplingBottleneck if ever fed odd dimensions directly.
"""


def _conv_bn_act(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.PReLU(out_ch),
    )


class Stem(nn.Module):
    """Two 3x3 convs, single channel in, `stem_channels` out."""

    def __init__(self, stem_channels: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_act(1, stem_channels),
            _conv_bn_act(stem_channels, stem_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallRefinementENet(nn.Module):
    def __init__(
        self,
        stem_channels: int = 8,
        stage_channels: int = 32,
        context_dilations: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        if stage_channels % 4 != 0:
            raise ValueError("stage_channels must be divisible by 4 (RegularBottleneck's internal_ratio=4).")

        cw = stage_channels
        self.stem_img = Stem(stem_channels)
        self.stem_pred = Stem(stem_channels)
        self.fuse_stems = nn.Sequential(
            nn.Conv2d(2 * stem_channels, cw, kernel_size=1, bias=False),
            nn.BatchNorm2d(cw),
            nn.PReLU(cw),
        )

        self.down = DownsamplingBottleneck(cw, 2 * cw, dropout_p=0.1)
        self.context = nn.Sequential(
            *[
                RegularBottleneck(2 * cw, padding=d, dilation=d, dropout_p=0.1)
                for d in context_dilations
            ]
        )
        self.up = UpsamplingBottleneck(2 * cw, cw)

        self.fuse_skip = nn.Sequential(
            nn.Conv2d(2 * cw, cw, kernel_size=1, bias=False),
            nn.BatchNorm2d(cw),
            nn.PReLU(cw),
        )
        self.refine = nn.Sequential(
            RegularBottleneck(cw, dropout_p=0.1),
            RegularBottleneck(cw, kernel_size=5, padding=2, asymmetric=True, dropout_p=0.1),
        )
        self.head = nn.Conv2d(cw, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 2:
            raise ValueError(f"SmallRefinementENet expects 2 input channels (image, predicted_mask), got {x.shape[1]}.")
        image, pred = x[:, 0:1], x[:, 1:2]

        fused_full = self.fuse_stems(torch.cat([self.stem_img(image), self.stem_pred(pred)], dim=1))

        down, indices, pre_pool_size = self.down(fused_full)
        ctx = self.context(down)
        up = self.up(ctx, pre_pool_size, indices)

        merged = self.fuse_skip(torch.cat([up, fused_full], dim=1))
        out = self.refine(merged)
        return self.head(out)
