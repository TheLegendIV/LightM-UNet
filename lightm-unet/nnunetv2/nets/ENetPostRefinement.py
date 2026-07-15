from __future__ import annotations

import torch
from torch import nn

from nnunetv2.nets.ENet import ENet, InitialBlock

"""
ENetPostRefinement -- the "separate stems, fuse later" answer to ENetPost's
"single conv, fuse immediately" design (nnUNetTrainerENetPost.py /
nnUNetTrainerENet.py's build_network_architecture just builds plain ENet
with in_channels=2, mixing both channels in ENet's stock InitialBlock's very
first conv). Same downstream architecture, same clDice-augmented loss
(nnUNetTrainerENetPostRefinement.py), same Dataset509_ARCADE_ENetPost -- the
only difference is what happens to the two input channels before down1.

Why this exists: Dataset509's channel 1 isn't an independent modality, it's
a *prior guess* (nnUNetTrainerSmallENet trained on Dataset507's own
reconstructed vessel probability). With early fusion, the first conv layer
has unrestricted, immediate access to that guess and can take the easy path
of leaning on it heavily -- the same passthrough-shortcut risk
nnUNetTrainerSmallRefinementENet needed a gap-focused loss to fight on
Dataset508 (nnUNetTrainerSmallRefinementENet.py's module docstring). Giving
each channel its own stem doesn't eliminate that risk, but it structurally
guarantees the raw-image pathway develops real independent features before
the probability channel gets a chance to dominate, instead of being
available to shortcut from pixel one.

    image (1ch) -> stem_a (ENet InitialBlock, 1->stem_channels, H/2 W/2) -\
                                                                             >-- concat (2*stem_channels) -> 1x1 fuse -> initial_channels, H/2 W/2
    prob  (1ch) -> stem_b (ENet InitialBlock, 1->stem_channels, H/2 W/2) -/
        -> (everything from here on is stock ENet: down1/regular1/down2/
            stage2/stage3/up4/regular4/up5/regular5/final, unchanged)

Each stem is a real InitialBlock (not a plain conv) -- same stride-2
conv+maxpool structure ENet's own single-channel InitialBlock uses -- so
each channel gets the exact same *kind* of low-level treatment ENetOriginal
would give a single input channel, just duplicated once per channel instead
of applied jointly. The fusion step (1x1 conv + BN + PReLU on the
concatenated stems) is the only place the two channels actually mix, mirroring
SmallRefinementENet's fuse_stems design at ENetOriginal's scale.
"""


class TwoStemInitialBlock(nn.Module):
    """Two-channel analogue of ENet's InitialBlock: each input channel gets
    its own InitialBlock stem before an explicit 1x1-conv fusion, instead of
    both channels being mixed together in a single shared first conv."""

    def __init__(self, stem_channels: int, out_channels: int):
        super().__init__()
        if stem_channels <= 1:
            raise ValueError("TwoStemInitialBlock stem_channels must exceed 1 (InitialBlock's own constraint).")
        self.stem_a = InitialBlock(1, stem_channels)
        self.stem_b = InitialBlock(1, stem_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * stem_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 2:
            raise ValueError(f"TwoStemInitialBlock expects exactly 2 input channels, got {x.shape[1]}.")
        a = self.stem_a(x[:, 0:1])
        b = self.stem_b(x[:, 1:2])
        return self.fuse(torch.cat([a, b], dim=1))


class ENetPostRefinement(ENet):
    """ENet with a two-stem initial block instead of the stock single-conv
    one -- always 2 input channels (image, upstream-model probability), same
    out_channels/channels convention as ENet otherwise."""

    def __init__(
        self,
        out_channels: int = 4,
        channels: tuple[int, int, int, int, int] = (20, 72, 144, 72, 20),
        stem_channels: int | None = None,
    ):
        # nn.Module.__setattr__ requires self._parameters etc. to already
        # exist, which only happens after nn.Module.__init__() runs (inside
        # ENet.__init__ -> super().__init__()) -- but _build_initial_block
        # (called from within that same ENet.__init__) needs stem_channels
        # available already. object.__setattr__ bypasses nn.Module's
        # bookkeeping for this one plain int/None value, which is safe here
        # since it's never a Parameter/Module/Tensor.
        object.__setattr__(self, "_stem_channels_override", stem_channels)
        super().__init__(in_channels=2, out_channels=out_channels, channels=channels)

    def _build_initial_block(self, in_channels: int, initial_channels: int) -> nn.Module:
        if in_channels != 2:
            raise ValueError(f"ENetPostRefinement requires exactly 2 input channels, got {in_channels}.")
        stem_channels = self._stem_channels_override or max(2, initial_channels // 2)
        return TwoStemInitialBlock(stem_channels, initial_channels)
