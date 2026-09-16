"""Export a FINN-compatible, PER-BLOCK-heterogeneous-bit-width mirror of the
REAL, QAT fine-tuned nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_
perblock checkpoint (12_separable_dense_relu_min4_ft50ep -- CHANNELS=
(4,16,32,16,4), separable_dilated=True, use_dsc=False, dense_dilation
context pattern, plain ReLU everywhere -- USE_PRELU=False), using the SAME
per-block HAWQ bit-width scheme (compression/hawq/block_bits_12_separable_
dense_relu_min4.json) as finn_export_12_separable_dense_relu_min4_hawq_dummy.py,
WITH REAL TRAINED WEIGHTS transferred wherever the FINN-safe topology is
structurally identical to CombinedQuantENet's real architecture.

Same "topology fix" fresh-init exceptions every other HAWQ export in this
repo documents:
  - down1/down2.shortcut_proj (NEW learned 1x1 projection -- real model's
    downsampling shortcut is parameter-free MaxPool+zero-pad).
  - up4/up5.main_up/main_bn (NEW learned conv-transpose replacing the real
    model's main_proj 1x1-conv + parameter-free bilinear/unpool).
  - every block's trailing activations + .act/.requant (QuantReLU/
    QuantEltwiseAdd -> plain ReLU/const-scale requant, different classes).
  - final.bias (real model has a trained bias; FINN-safe final layer
    requires bias=False for threshold streamlining).
`initial` IS transferred (real Concat-based FINNInitialBlockHAWQ, BN split
by channel index into conv_bn/pool_bn -- exact/lossless), same as every
other _hawq_real/_hawq_trained export in this repo.

The FINN-safe block classes (FINNInitialBlockHAWQ/FINNDownsamplingBottleneckHAWQ/
FINNUpsamplingBottleneckHAWQ/FINNRegularBottleneckHAWQ/
FINNRegularBottleneckSepDilatedHAWQ/FINNQuantENet12SepDenseReluHAWQ) are a
byte-for-byte copy of finn_export_12_separable_dense_relu_min4_hawq_dummy.py's
own -- only this file's checkpoint-loading + transfer_weights() are new.

REAL checkpoint: data/nnUNet_results/Dataset509_ARCADE_1x1_4c/
nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock_12_separable_
dense_relu_min4_ft50ep__nnUNetPlans__2d/fold_0/checkpoint_best.pth -- loaded
via CombinedQuantENet directly (same block_bits file + common_kwargs the
trainer itself uses, see enet/nnunetv2/training/nnUNetTrainer/
nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock.py).

Usage (run inside the pytorch training container):
    python hardware/finn_export_12_separable_dense_relu_min4_hawq_trained.py

Output: hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_min4_hawq_trained_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_min4_hawq_trained_int8.onnx \\
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnunetv2.nets.QuantENet import _quant_conv2d  # noqa: E402
from nnunetv2.nets.ENet import DENSE_DILATION_PATTERN  # noqa: E402
from nnunetv2.nets.CombinedQuantENet import CombinedQuantENet  # noqa: E402
from finn_export_s13_leaky_frozen import _plain_relu_factory, _requant_factory, _val, export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DEFAULT_BLOCK_BITS_FILE = REPO_ROOT / "compression" / "hawq" / "artifacts" / "block_bits_12_separable_dense_relu_min4.json"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock_12_separable_dense_relu_min4_ft50ep__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)


# ---------------------------------------------------------------------------
# Per-block-bit-width FINN-safe block definitions -- byte-for-byte copy of
# finn_export_12_separable_dense_relu_min4_hawq_dummy.py's own.
# ---------------------------------------------------------------------------

class FINNInitialBlockHAWQ(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int):
        super().__init__()
        self.conv_ch = out_ch - in_ch
        self.conv = _quant_conv2d(in_ch, self.conv_ch, weight_bits, kernel_size=3, stride=2, padding=1)
        self.conv_bn = nn.BatchNorm2d(self.conv_ch)
        self.conv_act = _plain_relu_factory(self.conv_ch, act_bits)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_bn = nn.BatchNorm2d(in_ch)
        self.pool_act = _plain_relu_factory(in_ch, act_bits)

        self.requant = _requant_factory(act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.requant(_val(self.conv_act(self.conv_bn(self.conv(x)))))
        pool_out = self.requant(_val(self.pool_act(self.pool_bn(self.pool(x)))))
        return torch.cat([conv_out, pool_out], dim=1)


class FINNDownsamplingBottleneckHAWQ(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int,
                 dropout_p: float = 0.01, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, out_ch // 4)

        self.shortcut_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.shortcut_proj = nn.Sequential(
            _quant_conv2d(in_ch, out_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _plain_relu_factory(out_ch, act_bits),
        )
        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, weight_bits, kernel_size=2, stride=2),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _plain_relu_factory(out_ch, act_bits),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = _plain_relu_factory(out_ch, act_bits)
        self.requant2 = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_proj(self.shortcut_pool(x)) if self.residual else None
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant2(_val(out)) + self.requant2(_val(shortcut))
        return self.act(out)


class FINNUpsamplingBottleneckHAWQ(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, in_ch // 4)

        self.main_up = qnn.QuantConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False,
            weight_bit_width=weight_bits, weight_quant=Int8WeightPerTensorFloat,
        )
        self.main_bn = nn.BatchNorm2d(out_ch)
        self.main_act = _plain_relu_factory(out_ch, act_bits)

        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.up = nn.Sequential(
            qnn.QuantConvTranspose2d(
                internal_ch, internal_ch, kernel_size=2, stride=2, bias=False,
                weight_bit_width=weight_bits, weight_quant=Int8WeightPerTensorFloat,
            ),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _plain_relu_factory(out_ch, act_bits),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.act = _plain_relu_factory(out_ch, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_act(self.main_bn(self.main_up(x)))
        out = self.dropout(self.expand(self.up(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(main))
        return self.act(out)


class FINNRegularBottleneckHAWQ(nn.Module):
    def __init__(self, channels: int, weight_bits: int, act_bits: int,
                 internal_ratio: int = 4, dropout_p: float = 0.1, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, weight_bits, kernel_size=1),
            nn.BatchNorm2d(channels),
            _plain_relu_factory(channels, act_bits),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = _plain_relu_factory(channels, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


class FINNRegularBottleneckSepDilatedHAWQ(nn.Module):
    def __init__(self, channels: int, weight_bits: int, act_bits: int, dilation: int,
                 internal_ratio: int = 4, kernel_size: int = 3, dropout_p: float = 0.1,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)
        padding = dilation

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=(kernel_size, 1),
                          padding=(padding, 0), dilation=dilation),
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=(1, kernel_size),
                          padding=(0, padding), dilation=dilation),
        )
        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_ch),
            _plain_relu_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, weight_bits, kernel_size=1),
            nn.BatchNorm2d(channels),
            _plain_relu_factory(channels, act_bits),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = _plain_relu_factory(channels, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.expand(self.conv_bn_act(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


def _make_shallow_stage(channels: int, n: int, block_weight_bits: dict, block_act_bits: dict,
                         dropout_p: float, name_prefix: str, residual: bool) -> nn.Sequential:
    blocks = []
    for i in range(n):
        name = f"{name_prefix}.{i}"
        blocks.append(FINNRegularBottleneckHAWQ(
            channels, block_weight_bits[name], block_act_bits[name], dropout_p=dropout_p, residual=residual,
        ))
    return nn.Sequential(*blocks)


def _make_context_stage(channels: int, n: int, block_weight_bits: dict, block_act_bits: dict,
                         name_prefix: str, residual: bool) -> nn.Sequential:
    pattern = DENSE_DILATION_PATTERN
    blocks = []
    for i in range(n):
        slot = pattern[i % len(pattern)]
        dilation = slot.get("dilation", 1)
        name = f"{name_prefix}.{i}"
        w, a = block_weight_bits[name], block_act_bits[name]
        if dilation != 1:
            blocks.append(FINNRegularBottleneckSepDilatedHAWQ(
                channels, w, a, dilation=dilation, dropout_p=0.1, residual=residual,
            ))
        else:
            blocks.append(FINNRegularBottleneckHAWQ(channels, w, a, dropout_p=0.1, residual=residual))
    return nn.Sequential(*blocks)


class FINNQuantENet12SepDenseReluHAWQ(nn.Module):
    def __init__(
        self, block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
        in_channels: int = 1, out_channels: int = 5,
        channels: tuple[int, ...] = CHANNELS, bottlenecks_per_stage: tuple[int, ...] = BOTTLENECKS_PER_STAGE,
        residual: bool = True,
    ):
        super().__init__()
        c0, c1, c23, c4, c5 = channels
        n1, n2, n3, n4, n5 = bottlenecks_per_stage

        self.initial = FINNInitialBlockHAWQ(in_channels, c0, block_weight_bits["initial"], block_act_bits["initial"])

        self.down1 = FINNDownsamplingBottleneckHAWQ(
            c0, c1, block_weight_bits["down1"], block_act_bits["down1"], dropout_p=0.01, residual=residual,
        )
        self.regular1 = _make_shallow_stage(c1, n1, block_weight_bits, block_act_bits, 0.01, "regular1", residual)

        self.down2 = FINNDownsamplingBottleneckHAWQ(
            c1, c23, block_weight_bits["down2"], block_act_bits["down2"], dropout_p=0.1, residual=residual,
        )
        self.stage2 = _make_context_stage(c23, n2, block_weight_bits, block_act_bits, "stage2", residual)
        self.stage3 = _make_context_stage(c23, n3, block_weight_bits, block_act_bits, "stage3", residual)

        self.up4 = FINNUpsamplingBottleneckHAWQ(c23, c4, block_weight_bits["up4"], block_act_bits["up4"], residual=residual)
        self.regular4 = _make_shallow_stage(c4, n4, block_weight_bits, block_act_bits, 0.1, "regular4", residual)

        self.up5 = FINNUpsamplingBottleneckHAWQ(c4, c5, block_weight_bits["up5"], block_act_bits["up5"], residual=residual)
        self.regular5 = _make_shallow_stage(c5, n5, block_weight_bits, block_act_bits, 0.1, "regular5", residual)

        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=False,
            weight_bit_width=block_weight_bits["final"], weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.regular1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.stage3(x)
        x = self.regular4(self.up4(x))
        x = self.regular5(self.up5(x))
        out = self.final(x)
        return out.value if hasattr(out, "value") else out


# ---------------------------------------------------------------------------
# Real-weight transfer -- reduce/expand/downsampling/upsampling/initial are
# all the same generic QuantENet.py primitives every other HAWQ export in
# this repo transfers the same way; the two REGULAR-bottleneck cases
# (plain dilation=1 single-conv vs separable_dilated (k,1)+(1,k)) are
# specific to this architecture (SEPARABLE_DILATED=True, USE_DSC=False --
# see QuantENet.py's QuantRegularBottleneck for the exact src structure
# both helpers below mirror).
# ---------------------------------------------------------------------------

def _copy_indices(dst_seq, src_seq, indices: list[int], note: str, report: list[str]) -> None:
    n = 0
    for i in indices:
        dst_seq[i].load_state_dict(src_seq[i].state_dict())
        n += sum(p.numel() for p in src_seq[i].parameters())
    report.append(f"  [OK]    {note}: {n} params transferred")


def _fresh(note: str, report: list[str]) -> None:
    report.append(f"  [FRESH] {note}")


def _transfer_initial(dst, src, name: str, report: list[str]) -> None:
    dst.conv.load_state_dict(src.conv.state_dict())
    report.append(f"  [OK]    {name}.conv weight: {src.conv.weight.numel()} params transferred")
    conv_ch = dst.conv_ch
    with torch.no_grad():
        dst.conv_bn.weight.copy_(src.bn.weight[:conv_ch])
        dst.conv_bn.bias.copy_(src.bn.bias[:conv_ch])
        dst.conv_bn.running_mean.copy_(src.bn.running_mean[:conv_ch])
        dst.conv_bn.running_var.copy_(src.bn.running_var[:conv_ch])
        dst.pool_bn.weight.copy_(src.bn.weight[conv_ch:])
        dst.pool_bn.bias.copy_(src.bn.bias[conv_ch:])
        dst.pool_bn.running_mean.copy_(src.bn.running_mean[conv_ch:])
        dst.pool_bn.running_var.copy_(src.bn.running_var[conv_ch:])
    report.append(f"  [OK]    {name}.bn split into conv_bn[0:{conv_ch}]/pool_bn[{conv_ch}:] "
                  f"(exact per-channel slice of the real trained BN, lossless)")
    _fresh(f"{name}.conv_act/.pool_act/.requant (trailing activations, fresh-init)", report)


def _transfer_downsampling(dst, src, name: str, report: list[str]) -> None:
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv, src.conv, [0, 1], f"{name}.conv (conv+BN)", report)
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}.shortcut_proj (NEW learned 1x1 projection -- real model's downsampling "
           f"shortcut is parameter-free MaxPool+zero-pad, no projection conv at all)", report)
    _fresh(f"{name}: reduce/conv/expand trailing activations + .act/.requant", report)


def _transfer_upsampling(dst, src, name: str, report: list[str]) -> None:
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.up, src.up, [0, 1], f"{name}.up (conv_transpose+BN)", report)
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}.main_up/main_bn (NEW learned conv-transpose replacing the real model's "
           f"main_proj 1x1-conv + parameter-free bilinear/unpool)", report)
    _fresh(f"{name}: reduce/up/expand trailing activations + .main_act/.act/.requant", report)


def _transfer_regular_plain(dst, src, name: str, report: list[str]) -> None:
    """src.conv (dilation=1) is a BARE conv module (not a Sequential) --
    src.conv_bn_act = Sequential(src.conv, BN, act) carries the matching BN.
    dst.conv = Sequential(conv, BN, act) -- copy conv<-src.conv,
    BN<-src.conv_bn_act[1]."""
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    dst.conv[0].load_state_dict(src.conv.state_dict())
    dst.conv[1].load_state_dict(src.conv_bn_act[1].state_dict())
    n = sum(p.numel() for p in src.conv.parameters()) + sum(p.numel() for p in src.conv_bn_act[1].parameters())
    report.append(f"  [OK]    {name}.conv (single 3x3 conv + BN, non-dilated): {n} params transferred")
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}: reduce/conv/expand trailing activations + .act/.requant", report)


def _transfer_regular_sepdilated(dst, src, name: str, report: list[str]) -> None:
    """src.conv (separable_dilated, dilation!=1) = Sequential(h_conv, BN,
    act, v_conv) -- structurally IDENTICAL to dst.conv, index-for-index
    (only the act at index 2 differs, fresh). src.conv_bn_act[1] (outer BN
    after v_conv) matches dst.conv_bn_act[1] the same way."""
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv, src.conv, [0, 1, 3], f"{name}.conv[0,1,3] (h_conv+BN, v_conv)", report)
    dst.conv_bn_act[1].load_state_dict(src.conv_bn_act[1].state_dict())
    report.append(f"  [OK]    {name}.conv_bn_act[1] (outer BN after v_conv): "
                  f"{sum(p.numel() for p in src.conv_bn_act[1].parameters())} params transferred")
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}: reduce/conv/expand trailing activations + .act/.requant", report)


def transfer_weights(dst: FINNQuantENet12SepDenseReluHAWQ, src: CombinedQuantENet) -> list[str]:
    report: list[str] = []

    _transfer_initial(dst.initial, src.initial, "initial", report)

    _transfer_downsampling(dst.down1, src.down1, "down1", report)
    for i, (d, s) in enumerate(zip(dst.regular1, src.regular1)):
        _transfer_regular_plain(d, s, f"regular1.{i}", report)

    _transfer_downsampling(dst.down2, src.down2, "down2", report)

    for stage_name, dst_stage, src_stage in (
        ("stage2", dst.stage2, src.stage2), ("stage3", dst.stage3, src.stage3),
    ):
        for i, (d, s) in enumerate(zip(dst_stage, src_stage)):
            dilation = DENSE_DILATION_PATTERN[i % len(DENSE_DILATION_PATTERN)].get("dilation", 1)
            if dilation != 1:
                _transfer_regular_sepdilated(d, s, f"{stage_name}.{i}", report)
            else:
                _transfer_regular_plain(d, s, f"{stage_name}.{i}", report)

    _transfer_upsampling(dst.up4, src.up4, "up4", report)
    for i, (d, s) in enumerate(zip(dst.regular4, src.regular4)):
        _transfer_regular_plain(d, s, f"regular4.{i}", report)

    _transfer_upsampling(dst.up5, src.up5, "up5", report)
    for i, (d, s) in enumerate(zip(dst.regular5, src.regular5)):
        _transfer_regular_plain(d, s, f"regular5.{i}", report)

    dst.final.weight.data.copy_(src.final.weight.data)
    report.append(f"  [OK]    final.weight (conv_transpose kernel): {src.final.weight.numel()} params transferred")
    _fresh("final.bias (real model has a trained bias; FINN-safe final layer requires "
           "bias=False for threshold streamlining)", report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--block-bits-file", default=str(DEFAULT_BLOCK_BITS_FILE))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--no-residuals", action="store_true")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    with open(args.block_bits_file) as f:
        block_bits = json.load(f)
    block_weight_bits = block_bits["stage_weight_bits"]
    block_act_bits = block_bits["stage_act_bits"]

    print("\n=== 1. Building + loading the REAL QAT fine-tuned CombinedQuantENet (12_separable_dense_relu_min4) ===")
    real_model = CombinedQuantENet(
        block_weight_bits, block_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels, channels=CHANNELS,
        bottlenecks_per_stage=BOTTLENECKS_PER_STAGE, context_pattern="dense_dilation",
        use_dilated=True, use_asymmetric=False, use_strided=True,
        use_dsc=False, dsc_no_projection=False, separable_dilated=True, trainable_slope=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["network_weights"]
    new_state_dict = {
        (key[7:] if key.startswith("module.") and key not in real_model.state_dict() else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = real_model.load_state_dict(new_state_dict, strict=True)
    real_model.eval()
    print(f"  Loaded {args.checkpoint} (epoch {checkpoint.get('current_epoch')}) -- missing={missing} unexpected={unexpected}")

    print("\n=== 2. Building the FINN-safe HAWQ mirror + transferring real weights ===")
    residual = not args.no_residuals
    model = FINNQuantENet12SepDenseReluHAWQ(
        block_weight_bits, block_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        residual=residual,
    )
    torch.manual_seed(0)
    report = transfer_weights(model, real_model)
    print("\n".join(report))
    model.eval()
    n_ok = sum(1 for line in report if "[OK]" in line)
    n_fresh = sum(1 for line in report if "[FRESH]" in line)
    print(f"\n  {n_ok} components transferred from the real checkpoint, {n_fresh} components fresh-initialized.")
    print(f"  weight_bits per block: {block_weight_bits}")
    print(f"  act_bits per block: {block_act_bits}")

    print("\n=== 3. Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    suffix = "_no_res" if not residual else ""
    name = f"quantEnet_12_separable_dense_relu_min4_hawq_trained_int8{suffix}"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
