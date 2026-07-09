"""ENet combo architectures — cross product of the winning single-axis ablations.

Two orthogonal axes are combined here:

  skip axis (from ENetSkip.py):
    E1 — residual addition at 4x only: Conv1x1(x_1x) -> ch_s4, added to x_4x
         (no concat, no fusion block, no 5x skip at all)
    A3 — raw skip at both 4x and 5x, Conv1x1 fusion (concat + 1x1 conv)

  context axis (from ENetGlobalCtx.py G3 / ENetUpscaleArch.py UAS2):
    G3      — stage 3 replaced by 4 directional PVMamba blocks
              (row-forward, row-backward, col-forward, col-backward)
    UAS2    — stage 2 extended from 8 to 10 dilated-conv blocks (+dil4, +dil8);
              stage 3 unchanged (standard 8-block dilated context)
    G3_UAS2 — both: stage 2 gets UAS2's +2-block extension, stage 3 is G3's
              4-block Mamba (UAS2's own +2-to-stage-3 doesn't apply here since
              G3 has already replaced stage 3's dilated-conv blocks entirely)

E1 and A3 both control the 4x injection point and cannot be combined with each
other, so they're run as two parallel combo families instead:

  id       skip  ctx
  EG1      E1    G3
  EG2      E1    UAS2
  EG3      E1    G3_UAS2
  AG1      A3    G3
  AG2      A3    UAS2
  AG3      A3    G3_UAS2

Fixed channel schedule for all combos: (20, 72, 144, 72, 20) — same as
ENetSkip / ENetGlobalCtx / ENetUpscaleArch, so params are directly comparable.
"""
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
from nnunetv2.nets.ENetGlobalCtx import PVMambaBlock

CHANNELS = (20, 72, 144, 72, 20)

SKIP_VARIANTS = frozenset({"E1", "A3"})
CTX_VARIANTS = frozenset({"G3", "UAS2", "G3_UAS2"})

_COMBO_TABLE: dict[str, tuple[str, str]] = {
    "EG1": ("E1", "G3"),
    "EG2": ("E1", "UAS2"),
    "EG3": ("E1", "G3_UAS2"),
    "AG1": ("A3", "G3"),
    "AG2": ("A3", "UAS2"),
    "AG3": ("A3", "G3_UAS2"),
}
VALID_EXPERIMENTS = frozenset(_COMBO_TABLE)


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def _ctx_stage(ch: int) -> nn.Sequential:
    """Standard 8-block dilated context stage (ENet baseline stage2/3)."""
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


def _ext_ctx_stage(ch: int) -> nn.Sequential:
    """UAS2's 10-block context stage: standard 8 + 2 extra (dil4, dil8)."""
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


def _mamba_stage3(ch: int) -> nn.ModuleList:
    """G3's stage 3: 4 directional PVMamba blocks (row-fwd, row-bwd, col-fwd, col-bwd)."""
    return nn.ModuleList([PVMambaBlock(ch, scan_dir=i) for i in range(4)])


def _dec_ref(ch: int) -> nn.Sequential:
    return nn.Sequential(RegularBottleneck(ch, relu=True), RegularBottleneck(ch, relu=True))


# ---------------------------------------------------------------------------
# ENetCombo
# ---------------------------------------------------------------------------

class ENetCombo(nn.Module):
    """ENet with a configurable skip strategy (E1/A3) and context strategy
    (G3/UAS2/G3_UAS2), combined along two independent axes."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, int, int, int, int] = CHANNELS,
        skip: str = "E1",
        ctx: str = "G3",
    ):
        super().__init__()
        if skip not in SKIP_VARIANTS:
            raise ValueError(f"Unknown skip variant '{skip}'. Valid: {sorted(SKIP_VARIANTS)}")
        if ctx not in CTX_VARIANTS:
            raise ValueError(f"Unknown ctx variant '{ctx}'. Valid: {sorted(CTX_VARIANTS)}")
        if len(channels) != 5:
            raise ValueError("ENetCombo expects 5 channel values: (initial, stage1, stage23, stage4, stage5).")
        ch0, ch1, ch23, ch4, ch5 = (int(c) for c in channels)
        if ch4 != ch1 or ch5 != ch0:
            raise ValueError(
                "ENetCombo MaxUnpool constraint: stage4 must equal stage1 and stage5 must equal initial. "
                f"Got: stage4={ch4}, stage1={ch1}, stage5={ch5}, initial={ch0}."
            )

        self.skip = skip
        self.ctx = ctx
        self.channels = (ch0, ch1, ch23, ch4, ch5)

        # --- Encoder ---
        self.initial = InitialBlock(in_channels, ch0)
        self.down1 = DownsamplingBottleneck(ch0, ch1, dropout_p=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(ch1, dropout_p=0.01) for _ in range(4)])
        self.down2 = DownsamplingBottleneck(ch1, ch23, dropout_p=0.1)

        self._mamba_stage3_active = ctx in ("G3", "G3_UAS2")
        self.stage2 = _ext_ctx_stage(ch23) if ctx in ("UAS2", "G3_UAS2") else _ctx_stage(ch23)
        self.stage3 = _mamba_stage3(ch23) if self._mamba_stage3_active else _ctx_stage(ch23)

        # --- Decoder ---
        self.up4 = UpsamplingBottleneck(ch23, ch4)
        self.regular4 = _dec_ref(ch4)
        self.up5 = UpsamplingBottleneck(ch4, ch5)
        self.regular5 = RegularBottleneck(ch5, dropout_p=0.1, relu=True)
        self.final = nn.ConvTranspose2d(ch5, out_channels, kernel_size=2, stride=2)

        # --- Skip modules ---
        self.proj_enc_4x: nn.Module | None = None
        self.fuse_4x: nn.Module | None = None
        self.fuse_5x: nn.Module | None = None
        if skip == "E1":
            self.proj_enc_4x = nn.Sequential(
                nn.Conv2d(ch1, ch4, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch4),
                nn.PReLU(ch4),
            )
        else:  # A3
            self.fuse_4x = nn.Sequential(
                nn.Conv2d(ch1 + ch4, ch4, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch4),
                nn.PReLU(ch4),
            )
            self.fuse_5x = nn.Sequential(
                nn.Conv2d(ch0 + ch5, ch5, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch5),
                nn.PReLU(ch5),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        x = self.initial(x)
        x_initial = x                          # ch0, H/2

        x, idx1, sz1 = self.down1(x)
        x = self.stage1(x)
        x_1x = x                               # ch1, H/4

        x, idx2, sz2 = self.down2(x)
        x = self.stage2(x)
        if self._mamba_stage3_active:
            for blk in self.stage3:
                x = blk(x)
        else:
            x = self.stage3(x)

        x = self.up4(x, sz2, idx2)             # ch23 -> ch4, H/8 -> H/4
        if self.skip == "E1":
            x = x + self.proj_enc_4x(x_1x)
        else:  # A3
            x = self.fuse_4x(torch.cat([x_1x, x], dim=1))
        x = self.regular4(x)

        x = self.up5(x, sz1, idx1)             # ch4 -> ch5, H/4 -> H/2
        if self.skip == "A3":
            x = self.fuse_5x(torch.cat([x_initial, x], dim=1))
        x = self.regular5(x)

        x = self.final(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def build_combo_model(
    experiment: str,
    in_channels: int = 1,
    out_channels: int = 4,
    channels: tuple = CHANNELS,
) -> nn.Module:
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(f"Unknown combo experiment '{experiment}'. Valid: {sorted(VALID_EXPERIMENTS)}")
    # NOTE: this line has been silently hardcoded to a single experiment ID
    # (ignoring the `experiment` argument entirely) at least twice by some
    # automated sync process -- _COMBO_TABLE["AG1"] from 2026-07-03 15:52 to
    # 2026-07-06 01:15, then _COMBO_TABLE["EG1"] from 2026-07-06 09:30 until
    # this fix. Either way, every requested experiment silently built
    # whichever architecture happened to be hardcoded, regardless of
    # COMBO_EXPERIMENT. See the self-test below, which asserts every
    # VALID_EXPERIMENTS entry actually gets its own distinct skip/ctx -- run
    # `python ENetCombo.py` before trusting any run's architecture again if
    # this line looks suspicious.
    skip, ctx = _COMBO_TABLE[experiment]
    return ENetCombo(
        in_channels=in_channels, out_channels=out_channels, channels=channels, skip=skip, ctx=ctx,
    )


if __name__ == "__main__":
    # Regression test for the "build_combo_model ignores `experiment`" bug:
    # every valid experiment ID must build a model whose actual skip/ctx
    # match _COMBO_TABLE, not just whichever one happens to be hardcoded.
    for exp, (expected_skip, expected_ctx) in _COMBO_TABLE.items():
        model = build_combo_model(exp, in_channels=1, out_channels=4, channels=CHANNELS)
        assert model.skip == expected_skip, (
            f"{exp}: expected skip={expected_skip!r}, got {model.skip!r} -- "
            f"build_combo_model is not respecting the `experiment` argument."
        )
        assert model.ctx == expected_ctx, (
            f"{exp}: expected ctx={expected_ctx!r}, got {model.ctx!r} -- "
            f"build_combo_model is not respecting the `experiment` argument."
        )
        has_proj_enc = model.proj_enc_4x is not None
        has_fuse = model.fuse_4x is not None
        expected_proj_enc = expected_skip == "E1"
        assert has_proj_enc == expected_proj_enc and has_fuse == (not expected_proj_enc), (
            f"{exp}: skip module mismatch (has_proj_enc={has_proj_enc}, has_fuse={has_fuse}) "
            f"for expected skip={expected_skip!r}."
        )
        print(f"{exp}: skip={model.skip} ctx={model.ctx} -- OK")
    print(f"All {len(_COMBO_TABLE)} experiment IDs build their own distinct architecture. build_combo_model is correct.")
