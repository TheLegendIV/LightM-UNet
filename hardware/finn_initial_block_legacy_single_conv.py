"""LEGACY REFERENCE ONLY -- not imported by any active script.

This is the pre-2026-08-28 `FINNInitialBlockHAWQ` used by every enet FINN
export script (finn_export_26_9_w24_hawq_joint_ptq.py,
finn_export_26_5_w24_hawq_joint.py, and their frozen historical siblings
s5/s13/s19): a single conv producing the FULL `out_ch` channels directly,
with no MaxPool branch and no Concat. It is topologically WRONG relative to
the real trained ENet InitialBlock (real: conv produces only
`out_ch - in_ch` channels, concatenated with a parameter-free MaxPool
branch over the raw input to reach `out_ch`) -- so `initial.conv`/`initial.
bn` could never be transferred from a real checkpoint (wrong shape) and had
to stay fresh-initialized every time.

As of 2026-08-28, the proper fix (real Concat-based initial block, see
FINNInitialBlockHAWQ in finn_export_26_9_w24_hawq_joint_ptq.py and
finn_export_26_5_w24_hawq_joint.py) is the default for all NEW enet FINN
export work -- it losslessly reuses the real trained initial.conv+bn (BN
split by channel index into per-branch affines, both branches merged via
FINN's StreamingConcat hardware op) instead of leaving that layer fresh.

This file only exists so the old shape is still available for reference/
comparison/rollback. Do not import it into any active pipeline.
"""
from __future__ import annotations

import torch
from torch import nn


class FINNInitialBlockHAWQLegacySingleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int, act_factory, quant_conv2d_fn):
        super().__init__()
        self.conv = quant_conv2d_fn(in_ch, out_ch, weight_bits, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_factory(out_ch, act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
