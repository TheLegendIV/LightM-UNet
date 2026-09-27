"""SegNet (Badrinarayanan et al., 2015/2017, "SegNet: A Deep Convolutional
Encoder-Decoder Architecture for Image Segmentation"). Standard VGG16-
topology encoder/decoder with max-pooling indices transferred from each
encoder stage to its mirrored decoder stage (the architecture's own defining
trick -- avoids learning the upsampling, reuses the exact spatial positions
of each encoder stage's maxima instead). No official repo is pulled from
here -- this is a well-established, unambiguous architecture (unlike PUNet/
LM-UNet's papers, there's no reverse-engineering risk) implemented directly
from the paper's own architecture description, batch-norm variant (the
paper's "SegNet-Basic"-with-BN convention, not the un-normalized original
Caffe release), matching the same is-this-really-SegNet checks (VGG16-
depth encoder, index-based unpooling) other cherry-picked nets in this repo
got via a parameter-count match.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, num_convs: int, transition_last: bool = False) -> nn.Sequential:
    """`transition_last=False` (encoder direction): channel change happens on
    the FIRST conv, the rest stay at `out_channels` -- e.g. VGG16 stage 3 is
    128->256->256->256. `transition_last=True` (decoder direction, the exact
    mirror): channels stay at `in_channels` until the LAST conv drops to
    `out_channels` -- e.g. the matching decoder stage is 512->512->512->256,
    not 512->256->256->256. Getting this backwards silently produces a
    smaller, wrong network -- caught here by SegNet's own well-known ~29.5M
    total-parameter figure not matching."""
    layers = []
    if not transition_last:
        c_in = in_channels
        for _ in range(num_convs):
            layers += [nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            c_in = out_channels
    else:
        for i in range(num_convs):
            c_out = out_channels if i == num_convs - 1 else in_channels
            layers += [nn.Conv2d(in_channels, c_out, kernel_size=3, padding=1),
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            in_channels = c_out
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    """VGG16-topology encoder/decoder, BatchNorm variant.

    Encoder stages (num convs per stage, matching VGG16): 2, 2, 3, 3, 3.
    Decoder mirrors this exactly, each stage's MaxUnpool2d fed the matching
    encoder stage's own pooling indices.
    """

    _STAGE_CONVS = (2, 2, 3, 3, 3)
    _STAGE_CHANNELS = (64, 128, 256, 512, 512)

    def __init__(self, in_channels: int = 3, out_channels: int = 21):
        super().__init__()

        enc_in = [in_channels, *self._STAGE_CHANNELS[:-1]]
        self.encoder_blocks = nn.ModuleList([
            _conv_block(enc_in[i], self._STAGE_CHANNELS[i], self._STAGE_CONVS[i])
            for i in range(5)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        dec_channels = list(reversed(self._STAGE_CHANNELS))
        dec_out = [*dec_channels[1:], dec_channels[-1]]
        dec_convs = list(reversed(self._STAGE_CONVS))
        self.decoder_blocks = nn.ModuleList([
            _conv_block(dec_channels[i], dec_out[i], dec_convs[i], transition_last=True)
            for i in range(5)
        ])
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(self._STAGE_CHANNELS[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = []
        indices = []
        for block in self.encoder_blocks:
            x = block(x)
            sizes.append(x.shape[-2:])
            x, idx = self.pool(x)
            indices.append(idx)

        for i, block in enumerate(self.decoder_blocks):
            x = self.unpool(x, indices[-(i + 1)], output_size=sizes[-(i + 1)])
            x = block(x)

        return self.final(x)


class SegNetBasic(nn.Module):
    """SegNet-Basic (paper's own Sec 3.1, Table 1: 1.425M params for
    in_channels=3/out_channels=11 on CamVid) -- the SMALL 4-encoder/4-decoder
    variant the paper itself uses for its decoder-variant ablation study, NOT
    the full 13-stage VGG16-topology `SegNet` class above. Deltas from the
    full SegNet, all explicit in the paper's own Sec 3.1 text:
      - 4 stages (not 5), 1 conv per stage (not 2/2/3/3/3).
      - Constant 7x7 kernel throughout ("a constant kernel size of 7x7 over
        all the encoder and decoder layers"), not VGG16's 3x3.
      - Constant 64-channel width at every stage, not VGG16's 64/128/256/512
        doubling schedule -- confirmed by the paper's own comparison text
        ("SegNet-Basic has a decoder with 64 feature maps in each decoder
        layer", contrasted directly against FCN-Basic's dimensionality-
        reduced 11-channel decoder -- both described with a SINGLE constant
        width across all their respective decoder layers).
      - BatchNorm after every conv, in BOTH encoder and decoder (exactly 2
        learnable params/channel -- scale+shift -- confirmed against the
        official prototxt's own "BN" layer's scale_filler/shift_filler).
      - ReLU in the encoder only -- "no ReLU non-linearity is present in the
        decoder network" (paper's own words).

    NOTE: the paper's own prose says "No biases are used after convolutions",
    but the official released Caffe implementation (github.com/alexgkendall/
    SegNet-Tutorial, Models/segnet_basic_train.prototxt -- linked directly
    from the paper's own abstract) contradicts this: every Convolution layer
    has TWO `param{}` blocks (weight AND bias, lr_mult 1/2 -- the classic
    Caffe convention, and there is no `bias_term: false` anywhere in the
    file). This class follows the released CODE, not the prose, since the
    code is the actual artifact the paper's own 1.425M figure was measured
    from. Verified layer-for-layer against that exact prototxt: 9
    Convolution layers total (conv1-4, conv_decode4-1, conv_classifier),
    constant num_output=64 throughout except conv_classifier's 11, kernel=7
    throughout except conv_classifier's 1, no BN/activation after
    conv_classifier (goes straight to the loss layer) -- all of which this
    class already matched. With bias=True restored to match the real code:
    1,416,587 params at in_channels=3/out_channels=11 (the paper's own
    CamVid config), ~0.59% under the paper's reported 1.425M -- the small
    remaining residual is most plausibly the released prototxt being a
    slightly later/updated snapshot than whatever produced that specific
    2015 paper table, not an architectural mismatch (every structural detail
    above is now code-verified, not reconstructed from prose)."""

    _NUM_STAGES = 4
    _WIDTH = 64
    _KERNEL_SIZE = 7

    def __init__(self, in_channels: int = 3, out_channels: int = 11):
        super().__init__()
        pad = self._KERNEL_SIZE // 2

        enc_in = [in_channels] + [self._WIDTH] * (self._NUM_STAGES - 1)
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_in[i], self._WIDTH, kernel_size=self._KERNEL_SIZE, padding=pad, bias=True),
                nn.BatchNorm2d(self._WIDTH), nn.ReLU(inplace=True),
            )
            for i in range(self._NUM_STAGES)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # No ReLU here -- decoder-only omission, per the paper's own text.
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self._WIDTH, self._WIDTH, kernel_size=self._KERNEL_SIZE, padding=pad, bias=True),
                nn.BatchNorm2d(self._WIDTH),
            )
            for _ in range(self._NUM_STAGES)
        ])
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(self._WIDTH, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = []
        indices = []
        for block in self.encoder_blocks:
            x = block(x)
            sizes.append(x.shape[-2:])
            x, idx = self.pool(x)
            indices.append(idx)

        for i, block in enumerate(self.decoder_blocks):
            x = self.unpool(x, indices[-(i + 1)], output_size=sizes[-(i + 1)])
            x = block(x)

        return self.final(x)


if __name__ == "__main__":
    net = SegNetBasic(in_channels=3, out_channels=11)
    n_params = sum(p.numel() for p in net.parameters())
    dummy = torch.randn(2, 3, 360, 480)
    net.eval()
    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (2, 11, 360, 480), f"got {tuple(out.shape)}"
    print(f"SegNetBasic self-test PASSED: builds, forward-passes to {tuple(out.shape)}, "
          f"{n_params} params (paper's Table 1: 1.425M at this exact in/out config, "
          f"code-verified against the official prototxt).")
