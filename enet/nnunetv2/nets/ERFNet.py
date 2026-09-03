"""ERFNet (Romera et al., 2017, IEEE T-ITS), implemented as a STANDALONE
model -- deliberately NOT a flag-composed variant of ENet.py's own ENet
class (unlike everything else in this nets/ directory), since ERFNet's
"non-bottleneck-1D" block is not a composition of ENet's existing
primitives (it has no 1x1 project/expand at all, a structurally different
residual shape from every ENet bottleneck -- see the block docstring
below).

Architecturally verified against the ORIGINAL author's Torch7 source
(github.com/Eromera/erfnet, cloned to this repo's own erfnet/ directory --
train/model/residual_modules.lua's own non_bt_1D/downsampler/upsampler
functions, plus encoder.lua/decoder.lua's own construction calls, read
line-by-line, not reconstructed from memory or a paraphrase of the paper
text). Confirmed real encoder/decoder layer sequence (encoder.lua calls
downsampler(3,16,...)/downsampler(16,64,drop/10,...)/non_bt_1D x5 (64,
drop/10,...)/downsampler(64,128,drop,...)/non_bt_1D x8 (128,drop,...
dilated=2,4,8,16 x2), THEN a temporary SpatialConvolution(128,#classes,1,1)
classifier -- decoder.lua's own `model:remove(#model.modules)` strips that
classifier back off before attaching upsampler(128,64)/non_bt_1D x2 (64,0,
...)/upsampler(64,16)/non_bt_1D x2 (16,0,...)/SpatialFullConvolution(16,
#classes,2,2,2,2) -- so the encoder's own temporary 1x1 classifier is
training-pretraining scaffolding ONLY, never part of the full network,
correctly excluded here):
    DownsamplerBlock(3 -> 16, dropout=0)
    DownsamplerBlock(16 -> 64, dropout=0.03)
    5x non_bottleneck_1d(64, dropout=0.03, dilation=1)
    DownsamplerBlock(64 -> 128, dropout=0.3)
    8x non_bottleneck_1d(128, dropout=0.3, dilation=2,4,8,16,2,4,8,16)
    UpsamplerBlock(128 -> 64)                        # no dropout anywhere in upsampler()
    2x non_bottleneck_1d(64, dropout=0, dilation=1)
    UpsamplerBlock(64 -> 16)
    2x non_bottleneck_1d(16, dropout=0, dilation=1)
    ConvTranspose2d(16 -> num_classes, kernel=2, stride=2)   # bare, no BN/act
Also confirmed line-by-line: BatchNorm2d(eps=1e-3) throughout (not
PyTorch's 1e-5 default -- cudnn.SpatialBatchNormalization(_, 1e-3) at
every call site in residual_modules.lua, not a typo if you diff against
ENet.py's own BatchNorm2d() calls); bias=True on every conv/deconv that
ISN'T immediately followed by a BatchNorm (Torch7's SpatialConvolution/
SpatialFullConvolution carry a bias term by default, and residual_modules.
lua never calls :noBias() anywhere); DownsamplerBlock's own dropout sits
BETWEEN its BatchNorm and its final ReLU (`modul:add(BN) ...
modul:add(Dropout) ... modul:add(ReLU)`), not after the activation; and
non_bottleneck_1d's own dilation applies ONLY to its second (3,1)+(1,3)
pair (steps 3-4), never its first (steps 1-2), regardless of the block's
own `dilated` argument.

Channels below use THIS REPO's own 5-value convention (initial, stage1,
context, stage4, stage5 -- see ENet.py's own `channels` docstring), not the
paper's fixed 3/16/64/128 -- e.g. U4-style channels=(4,16,32,16,4) plugs in
directly for an apples-to-apples channel-BUDGET comparison against every
ENet variant in this repo trained at the same width. Block COUNTS and the
dilation SCHEDULE are the real architecture's own, unchanged by the width
substitution (this is a channel-width delta, not a depth delta).

Deltas from ENet, all deliberate (see this file's own design doc for the
full rationale -- not reproduced here):
  - No 1x1 project/expand anywhere -- non_bottleneck_1d preserves channel
    width C end-to-end, so its own residual is a bare identity add (no
    projection needed on the skip either).
  - DownsamplerBlock (conv-stride-2 concatenated with maxpool-stride-2,
    ENet's own InitialBlock primitive) is reused at EVERY downsample stage,
    not just the first.
  - Decoder uses plain ConvTranspose2d (stride 2) at each upsample stage,
    NOT max-unpooling -- no pooling indices carried from encoder to
    decoder at all (this class's own forward() never touches indices).
  - Plain ReLU throughout -- no PReLU anywhere.
  - No long-range encoder<->decoder skip connections (tested and dropped
    by the original authors).
Training-time specifics (Adam/batch-12/momentum-0.9/wd-2e-4, LR 5e-4
halved-on-plateau, two-stage encoder-then-decoder training, horizontal-
flip+translation augmentation) are OUT OF SCOPE here by request -- this
repo trains every architecture (ERFNet included) through its own existing
single-stage AdamW/PolyLR pipeline (nnUNetTrainerENet's own configure_
optimizers) for a controlled architecture-only comparison, not the paper's
own training recipe.
"""
from __future__ import annotations

import torch
from torch import nn

# ERFNet's own BatchNorm eps -- PyTorch's default is 1e-5; the reference
# implementation uses 1e-3 throughout (DownsamplerBlock, UpsamplerBlock,
# and both BNs inside every non_bottleneck_1d). Kept as one named constant
# rather than a literal repeated at every BatchNorm2d call site.
_ERFNET_BN_EPS = 1e-3


class DownsamplerBlock(nn.Module):
    """Same primitive as ENet.py's own InitialBlock (conv 3x3 stride 2,
    channel count `out_channels - in_channels`, concatenated with a
    parallel maxpool-stride-2 branch, giving `out_channels` total) --
    ERFNet's own delta is reusing this block at EVERY downsample stage
    (3 total: in_channels->initial, initial->stage1, stage1->context),
    not just the first, unlike ENet's use_strided DownsamplingBottleneck
    (a real bottleneck with its own 1x1 projection) for its 2nd/3rd
    downsamples. bias=True on the strided conv (matches the reference --
    NOT ENet's own bias=False-everywhere convention).

    dropout_p: confirmed against the real Torch7 source (erfnet/train/
    model/residual_modules.lua's own `downsampler` function) -- BN then
    Dropout2d then ReLU, in that order (dropout sits BEFORE the
    activation, not after). The reference calls this with 0 at the first
    downsampler (in_channels->initial, a no-op), drop/10=0.03 at the
    second (initial->stage1, same rate non_bottleneck_1d's own stage1
    blocks use), and drop=0.3 at the third (stage1->context, same rate
    the context blocks use) -- see ERFNet's own constructor, which reuses
    stage1_dropout/context_dropout for exactly this."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("DownsamplerBlock out_channels must exceed in_channels.")
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, eps=_ERFNET_BN_EPS)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1))
        out = self.dropout(out)
        return self.act(out)


class NonBottleneck1D(nn.Module):
    """ERFNet's "non-bt-1D" block (the paper's own `non_bottleneck_1d`) --
    the core delta from every ENet bottleneck: NO 1x1 project/expand at all,
    channel width C preserved end-to-end, so the residual add is a bare
    identity (no projection needed on the skip). Two chained "factorized
    3x3" units (a plain one, then a possibly-dilated one), each unit itself
    a (3,1)->(1,3) pair with a real nonlinearity between the two 1D convs
    (not just a rank-1 weight factoring):
        1. 3x1 conv (bias=True)                  -> ReLU
        2. 1x3 conv (bias=True) -> BatchNorm      -> ReLU
        3. 3x1 conv (bias=True, dilated)          -> ReLU
        4. 1x3 conv (bias=True, dilated, same rate as 3) -> BatchNorm
        5. Dropout2d(p=dropout_p)
        6. + residual (identity)
        7. ReLU
    dropout_p varies by where this block sits in the real architecture
    (0.03 at the plain stage1 width, 0.3 at the dilated context width, 0 in
    the decoder -- see ERFNet's own constructor, which threads the right
    value to each call site; NOT a uniform 0.3 everywhere despite that
    being a common simplification of the paper text)."""

    def __init__(self, channels: int, dilation: int = 1, dropout_p: float = 0.0):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(channels, eps=_ERFNET_BN_EPS)
        self.conv3x1_2 = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), stride=1, padding=(dilation, 0), bias=True, dilation=(dilation, 1),
        )
        self.conv1x3_2 = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), stride=1, padding=(0, dilation), bias=True, dilation=(1, dilation),
        )
        self.bn2 = nn.BatchNorm2d(channels, eps=_ERFNET_BN_EPS)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv3x1_1(x))
        out = self.act(self.bn1(self.conv1x3_1(out)))
        out = self.act(self.conv3x1_2(out))
        out = self.bn2(self.conv1x3_2(out))
        out = self.dropout(out)
        return self.act(out + x)


class UpsamplerBlock(nn.Module):
    """Plain transposed-conv upsample-by-2, BatchNorm, ReLU -- ERFNet's own
    decoder stage primitive, deliberately NOT max-unpooling (unlike ENet's
    UpsamplingBottleneck under decoder_type="max_unpool"): no pooling
    indices are produced or consumed anywhere in this file."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=_ERFNET_BN_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# Real dilation schedule at the context (deepest) stage -- (2,4,8,16)
# repeated twice for the reference's own 8-block depth. Cycled via
# `i % len(_CONTEXT_DILATION_SCHEDULE)` for any other context_depth, same
# convention ENet.py's own DENSE_DILATION_PATTERN cycling uses.
_CONTEXT_DILATION_SCHEDULE: tuple[int, ...] = (2, 4, 8, 16)


class ERFNet(nn.Module):
    """channels: 5-value (initial, stage1, context, stage4, stage5) --
    SAME convention ENet.py's own `channels` param uses (stage2/stage3 are
    NOT split here -- the real ERFNet has one continuous context stage, not
    ENet's own down2->stage2->proj2_to_3->stage3 split), so e.g. U4-style
    (4,16,32,16,4) plugs in directly. stage1_depth/context_depth/
    decoder_depth default to the real architecture's own block counts
    (5/8/2) -- override only for an explicit depth-ablation, not for a
    width comparison (that's what `channels` is for)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 20,
        channels: tuple[int, int, int, int, int] = (16, 64, 128, 64, 16),
        stage1_depth: int = 5,
        context_depth: int = 8,
        decoder_depth: int = 2,
        stage1_dropout: float = 0.03,
        context_dropout: float = 0.3,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(
                "ERFNet expects five channel values (initial, stage1, context, stage4, stage5) -- "
                "see ENet.py's own `channels` docstring for the same convention (stage2/stage3 are "
                "NOT split here, unlike ENet -- ERFNet has one continuous context stage)."
            )
        initial_channels, stage1_channels, context_channels, stage4_channels, stage5_channels = channels
        if initial_channels <= in_channels:
            raise ValueError("ERFNet initial channels must exceed in_channels.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(int(c) for c in channels)

        # -- Encoder --------------------------------------------------------
        # Downsampler dropout: 0 at the first (reference's own literal 0,
        # a no-op), stage1_dropout at the second, context_dropout at the
        # third -- reusing the SAME two rates the following non_bottleneck_1d
        # stages use, exactly matching the reference (see DownsamplerBlock's
        # own docstring).
        self.down1 = DownsamplerBlock(in_channels, initial_channels, dropout_p=0.0)
        self.down2 = DownsamplerBlock(initial_channels, stage1_channels, dropout_p=stage1_dropout)
        self.stage1 = nn.Sequential(*[
            NonBottleneck1D(stage1_channels, dilation=1, dropout_p=stage1_dropout) for _ in range(stage1_depth)
        ])
        self.down3 = DownsamplerBlock(stage1_channels, context_channels, dropout_p=context_dropout)
        self.context = nn.Sequential(*[
            NonBottleneck1D(
                context_channels,
                dilation=_CONTEXT_DILATION_SCHEDULE[i % len(_CONTEXT_DILATION_SCHEDULE)],
                dropout_p=context_dropout,
            )
            for i in range(context_depth)
        ])

        # -- Decoder ----------------------------------------------------------
        self.up1 = UpsamplerBlock(context_channels, stage4_channels)
        self.regular4 = nn.Sequential(*[
            NonBottleneck1D(stage4_channels, dilation=1, dropout_p=decoder_dropout) for _ in range(decoder_depth)
        ])
        self.up2 = UpsamplerBlock(stage4_channels, stage5_channels)
        self.regular5 = nn.Sequential(*[
            NonBottleneck1D(stage5_channels, dilation=1, dropout_p=decoder_dropout) for _ in range(decoder_depth)
        ])
        # Bare transposed conv, no BN/activation -- same convention ENet.py's
        # own `final` layer already uses (a raw logits projection, not
        # another feature-refining block).
        self.final = nn.ConvTranspose2d(stage5_channels, out_channels, kernel_size=2, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.stage1(x)
        x = self.down3(x)
        x = self.context(x)
        x = self.up1(x)
        x = self.regular4(x)
        x = self.up2(x)
        x = self.regular5(x)
        return self.final(x)


if __name__ == "__main__":
    # Self-test, matching ENet.py's own convention: build + forward-pass at
    # 512x512, verify shapes, and verify the dilation schedule/dropout
    # rates landed exactly where the reference puts them.
    dummy = torch.randn(1, 1, 512, 512)

    U4_CHANNELS = (4, 16, 32, 16, 4)
    net = ERFNet(in_channels=1, out_channels=5, channels=U4_CHANNELS).eval()
    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (1, 5, 512, 512), f"U4: got {tuple(out.shape)}"
    assert len(net.stage1) == 5 and len(net.context) == 8 and len(net.regular4) == 2 and len(net.regular5) == 2, (
        "U4: block counts don't match the reference (5/8/2/2)."
    )
    expected_dilations = [2, 4, 8, 16, 2, 4, 8, 16]
    for i, (block, expected) in enumerate(zip(net.context, expected_dilations)):
        got = block.conv3x1_2.dilation[0]
        assert got == expected, f"U4: context[{i}] expected dilation={expected}, got {got}"
        assert block.dropout.p == 0.3, f"U4: context[{i}] expected dropout=0.3, got {block.dropout.p}"
    for i, block in enumerate(net.stage1):
        assert block.conv3x1_2.dilation[0] == 1, f"U4: stage1[{i}] expected dilation=1"
        assert block.dropout.p == 0.03, f"U4: stage1[{i}] expected dropout=0.03, got {block.dropout.p}"
    for stage_name in ("regular4", "regular5"):
        for i, block in enumerate(getattr(net, stage_name)):
            assert block.dropout.p == 0.0, f"U4: {stage_name}[{i}] expected dropout=0.0, got {block.dropout.p}"
    # DownsamplerBlock dropout, confirmed against the real Torch7 source
    # (encoder.lua): 0 / stage1_dropout / context_dropout at the 1st/2nd/3rd
    # downsampler respectively.
    assert net.down1.dropout.p == 0.0, f"U4: down1 expected dropout=0.0, got {net.down1.dropout.p}"
    assert net.down2.dropout.p == 0.03, f"U4: down2 expected dropout=0.03, got {net.down2.dropout.p}"
    assert net.down3.dropout.p == 0.3, f"U4: down3 expected dropout=0.3, got {net.down3.dropout.p}"

    n_params = sum(p.numel() for p in net.parameters())
    print(f"ERFNet self-test PASSED: U4-width (channels={U4_CHANNELS}) builds, forward-passes to "
          f"{tuple(out.shape)}, block counts (5/8/2/2) and dilation schedule (2,4,8,16)x2 verified, "
          f"dropout rates (0.03/0.3/0.0, including all 3 DownsamplerBlocks) verified at their correct "
          f"stages. {n_params} params.")

    # Non-default channel widths still build (channel-agnostic, same shape
    # contract as ENet.py's own multi-width self-test).
    for channels in ((16, 64, 128, 64, 16), (8, 32, 64, 32, 8)):
        probe = ERFNet(in_channels=1, out_channels=5, channels=channels).eval()
        with torch.no_grad():
            probe_out = probe(dummy)
        assert probe_out.shape == (1, 5, 512, 512), f"channels={channels}: got {tuple(probe_out.shape)}"
    print("ERFNet multi-width self-test PASSED: paper-native (16,64,128,64,16) and an intermediate width both build + forward-pass correctly.")
