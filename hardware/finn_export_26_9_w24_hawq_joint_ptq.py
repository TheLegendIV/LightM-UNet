"""Export the REAL PTQ-calibrated per-block HAWQ 26_9_w24_s14w12_nonneg_block
checkpoint (nnUNetTrainerENetQuant_26_9_w24_s14w12_nonneg_block_ptq_block,
checkpoint_best.pth -- produced by compression/post-quantization/ptq_block.py
from the real trained FP32 nnUNetTrainerENet_26_9_w24_s14w12_nonneg_block
checkpoint, calibrated on real preprocessed images using
compression/hawq/block_bits_26_9_w24_s14w12_nonneg_block_acc1x_joint.json's
per-block (weight_bits, act_bits) and the REAL trained nonneg_block scalar
slopes in compression/post-quantization/slope_maps/
26_9_w24_s14w12_nonneg_block.json) through the FINN-safe topology, WITH REAL
CALIBRATED WEIGHTS transferred wherever the FINN-safe topology is
structurally identical to QuantENet26_9_w24_s14w12_nonneg_block's real
architecture -- same convention as finn_export_s19_hawq_block_trained.py
(see that file's own docstring for the FINN-topology deviations that stay
fresh-initialized: downsampling shortcut_proj, upsampling main_up/main_bn,
final bias -- NOTE: unlike that file, the initial block's conv+bn ARE now
transferred here, via a real Concat-based initial block that mirrors the
real model's conv+MaxPool+concat+BN structure -- see FINNInitialBlockHAWQ's
own docstring).

UNLIKE finn_export_26_5_w24_hawq_joint.py (fresh/random weights, resource-
probe only), this is a real deployable/accuracy-relevant artifact: the
slope map is the model's own REAL trained per-block nonneg_block scalar
(not a post-hoc per-channel average like 26_5_w24's "standard" variant), so
DecomposedLeakyAct here carries the actual trained activation shape, not
just a placeholder.

The FINN-safe block classes (FINNInitialBlockHAWQ/FINNDownsamplingBottleneckHAWQ/
FINNUpsamplingBottleneckHAWQ/FINNRegularBottleneckHAWQ/
FINNRegularBottleneckSepDilatedHAWQ/FINNQuantENet26_9_w24HAWQ) are structurally
IDENTICAL to finn_export_26_5_w24_hawq_joint.py's own (just a different
default CHANNELS tuple) -- copied here (not imported), same "don't import a
__main__-driven sibling script" convention every export script in this repo
follows. transfer_weights()'s per-block helpers are the dense_dilation-
pattern generalization of finn_export_s19_double_mid.py's own (every
stage2/stage3 slot here is separable-dilated -- DENSE_DILATION_PATTERN has
no dilation=1 slots, unlike S19's reg-interleaved-double-mid pattern).

IMPORTANT (build-side, not this script): same _fixup_degenerate_signed_bias
requirement as finn_export_26_5_w24_hawq_joint.py -- must run between
step_enet_streamline and step_enet_convert_to_hw.

Usage (run inside the pytorch training container, e.g. `lightm_pytorch`):
    python hardware/finn_export_26_9_w24_hawq_joint_ptq.py

Output: hardware/outputs/finn_exports/quantEnet_26_9_w24_hawq_joint_ptq_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_26_9_w24_hawq_joint_ptq_int8.onnx \\
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
from nnunetv2.nets.QuantENet26_9_w24_s14w12_nonneg_block import (  # noqa: E402
    BLOCK_NAMES, QuantENet26_9_w24_s14w12_nonneg_block,
)
from finn_export_s13_leaky_frozen import (  # noqa: E402
    _make_act_factory, _plain_relu_factory, _requant_factory, _val, export_model,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 12, 24, 12, 4)          # initial, s1, s23 (shared), s4, s5 -- 26_9_w24_s14w12_nonneg_block's own
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DEFAULT_SLOPE_MAP_FILE = (
    REPO_ROOT / "compression" / "post-quantization" / "slope_maps" / "26_9_w24_s14w12_nonneg_block.json"
)
DEFAULT_BLOCK_BITS_FILE = (
    REPO_ROOT / "compression" / "hawq" / "block_bits_26_9_w24_s14w12_nonneg_block_acc1x_joint.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data" / "nnUNet_results" / "Dataset509_ARCADE_1x1_4c"
    / "nnUNetTrainerENetQuant_26_9_w24_s14w12_nonneg_block_ptq_block__nnUNetPlans__2d"
    / "fold_0" / "checkpoint_best.pth"
)


# ---------------------------------------------------------------------------
# Per-block-bit-width FINN-safe block definitions -- structurally identical
# to finn_export_26_5_w24_hawq_joint.py's own (see that file for the
# canonical version).
# ---------------------------------------------------------------------------

class FINNInitialBlockHAWQ(nn.Module):
    """Structurally identical to the real ENet InitialBlock: a learned conv
    branch (in_ch -> out_ch-in_ch) and a parameter-free MaxPool branch
    (in_ch -> in_ch) merged via FINN's StreamingConcat -- FINN's Concat HW op
    only supports the last/channel axis (finn.custom_op.fpgadataflow.concat.
    StreamingConcat), which is exactly the axis torch.cat(dim=1) uses here,
    so this is directly buildable (unlike the single full-out_ch conv this
    class used before, which could never reuse the real trained initial.conv
    -- wrong shape, always fresh-initialized).

    The real model's single BN(out_ch) (applied AFTER the concat) is split
    by channel index into two per-branch affines -- BN is per-channel, so
    slicing the trained weight/bias/running stats by index is an exact,
    lossless transfer, not an approximation. Each branch then needs its own
    trailing activation quantizer BEFORE the concat (FINN's InferConcatLayer
    requires every concat input to already be a coherent quantized integer
    type -- a raw pre-BN conv accumulator or a plain float can't be a concat
    input), then both are forced onto the SAME const-scale requant grid
    (the same self.requant-shared-instance trick already used at every
    residual Add in this file) immediately before the concat -- FINN's
    StreamingConcat has no per-stream rescale, it just concatenates raw
    integers, so both halves must already share one scale."""

    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int, act_factory):
        super().__init__()
        self.conv_ch = out_ch - in_ch
        self.conv = _quant_conv2d(in_ch, self.conv_ch, weight_bits, kernel_size=3, stride=2, padding=1)
        self.conv_bn = nn.BatchNorm2d(self.conv_ch)
        self.conv_act = act_factory(self.conv_ch, act_bits)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_bn = nn.BatchNorm2d(in_ch)
        self.pool_act = act_factory(in_ch, act_bits)

        self.requant = _requant_factory(act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.requant(_val(self.conv_act(self.conv_bn(self.conv(x)))))
        pool_out = self.requant(_val(self.pool_act(self.pool_bn(self.pool(x)))))
        return torch.cat([conv_out, pool_out], dim=1)


class FINNDownsamplingBottleneckHAWQ(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int, act_factory,
                 dropout_p: float = 0.01, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, out_ch // 4)

        self.shortcut_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.shortcut_proj = nn.Sequential(
            _quant_conv2d(in_ch, out_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            act_factory(out_ch, act_bits),
        )
        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, weight_bits, kernel_size=2, stride=2),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            act_factory(out_ch, act_bits),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(out_ch, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_proj(self.shortcut_pool(x)) if self.residual else None
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(shortcut))
        return self.act(out)


class FINNUpsamplingBottleneckHAWQ(nn.Module):
    """Decoder-half block: always plain QuantReLU (never leaky)."""

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
    """Plain (non-dilated, non-separable) regular bottleneck -- regular1/
    regular4/regular5 (dilation=1 always)."""

    def __init__(self, channels: int, weight_bits: int, act_bits: int, act_factory,
                 internal_ratio: int = 4, dropout_p: float = 0.1, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, weight_bits, kernel_size=1),
            nn.BatchNorm2d(channels),
            act_factory(channels, act_bits),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(channels, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


class FINNRegularBottleneckSepDilatedHAWQ(nn.Module):
    """Dilated context-stage bottleneck, (k,1)+(1,k) separable-dilated
    factoring -- every stage2/stage3 slot (DENSE_DILATION_PATTERN has no
    plain, dilation=1 slots at all)."""

    def __init__(self, channels: int, weight_bits: int, act_bits: int, act_factory, dilation: int,
                 internal_ratio: int = 4, kernel_size: int = 3, dropout_p: float = 0.1,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)
        padding = dilation

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, weight_bits, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=(kernel_size, 1),
                          padding=(padding, 0), dilation=dilation),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
            _quant_conv2d(internal_ch, internal_ch, weight_bits, kernel_size=(1, kernel_size),
                          padding=(0, padding), dilation=dilation),
        )
        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, act_bits),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, weight_bits, kernel_size=1),
            nn.BatchNorm2d(channels),
            act_factory(channels, act_bits),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(channels, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.dropout(self.expand(out))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


# ---------------------------------------------------------------------------
# Model assembly -- same topology as QuantENet26_9_w24_s14w12_nonneg_block,
# just every block looks up its own (weight_bits, act_bits) pair by
# BLOCK_NAMES key, and encoder blocks get a FROZEN per-block leaky slope
# (DecomposedLeakyAct) instead of plain QuantReLU.
# ---------------------------------------------------------------------------

def _make_shallow_stage(channels: int, n: int, block_weight_bits: dict, block_act_bits: dict,
                         dropout_p: float, name_prefix: str, residual: bool,
                         slope_map: dict) -> nn.Sequential:
    blocks = []
    for i in range(n):
        name = f"{name_prefix}.{i}"
        blocks.append(FINNRegularBottleneckHAWQ(
            channels, block_weight_bits[name], block_act_bits[name], _make_act_factory(slope_map.get(name)),
            dropout_p=dropout_p, residual=residual,
        ))
    return nn.Sequential(*blocks)


def _make_context_stage(channels: int, n: int, block_weight_bits: dict, block_act_bits: dict,
                         name_prefix: str, residual: bool, slope_map: dict) -> nn.Sequential:
    pattern = DENSE_DILATION_PATTERN
    blocks = []
    for i in range(n):
        slot = pattern[i % len(pattern)]
        dilation = slot.get("dilation", 1)
        name = f"{name_prefix}.{i}"
        w, a = block_weight_bits[name], block_act_bits[name]
        act_factory = _make_act_factory(slope_map.get(name))
        if dilation != 1:
            blocks.append(FINNRegularBottleneckSepDilatedHAWQ(
                channels, w, a, act_factory, dilation=dilation, dropout_p=0.1, residual=residual,
            ))
        else:
            blocks.append(FINNRegularBottleneckHAWQ(channels, w, a, act_factory, dropout_p=0.1, residual=residual))
    return nn.Sequential(*blocks)


class FINNQuantENet26_9_w24HAWQ(nn.Module):
    """FINN-compatible, per-BLOCK-bit-width mirror of
    QuantENet26_9_w24_s14w12_nonneg_block."""

    def __init__(
        self, block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
        in_channels: int = 1, out_channels: int = 5,
        channels: tuple[int, ...] = CHANNELS, bottlenecks_per_stage: tuple[int, ...] = BOTTLENECKS_PER_STAGE,
        residual: bool = True, leaky_slope_map: dict | None = None,
    ):
        super().__init__()
        missing_w = [b for b in BLOCK_NAMES if b not in block_weight_bits]
        missing_a = [b for b in BLOCK_NAMES if b not in block_act_bits]
        if missing_w or missing_a:
            raise ValueError(
                f"block_weight_bits/block_act_bits must have one entry per {len(BLOCK_NAMES)} BLOCK_NAMES -- "
                f"missing weight keys: {missing_w}, missing act keys: {missing_a}."
            )
        c0, c1, c23, c4, c5 = channels
        n1, n2, n3, n4, n5 = bottlenecks_per_stage
        slope_map = leaky_slope_map or {}

        self.initial = FINNInitialBlockHAWQ(
            in_channels, c0, block_weight_bits["initial"], block_act_bits["initial"],
            _make_act_factory(slope_map.get("initial")),
        )

        self.down1 = FINNDownsamplingBottleneckHAWQ(
            c0, c1, block_weight_bits["down1"], block_act_bits["down1"], _make_act_factory(slope_map.get("down1")),
            dropout_p=0.01, residual=residual,
        )
        self.regular1 = _make_shallow_stage(c1, n1, block_weight_bits, block_act_bits, 0.01, "regular1", residual, slope_map)

        self.down2 = FINNDownsamplingBottleneckHAWQ(
            c1, c23, block_weight_bits["down2"], block_act_bits["down2"], _make_act_factory(slope_map.get("down2")),
            dropout_p=0.1, residual=residual,
        )
        self.stage2 = _make_context_stage(c23, n2, block_weight_bits, block_act_bits, "stage2", residual, slope_map)
        self.stage3 = _make_context_stage(c23, n3, block_weight_bits, block_act_bits, "stage3", residual, slope_map)

        # Decoder: always plain ReLU, regardless of leaky_slope_map.
        self.up4 = FINNUpsamplingBottleneckHAWQ(c23, c4, block_weight_bits["up4"], block_act_bits["up4"], residual=residual)
        self.regular4 = _make_shallow_stage(c4, n4, block_weight_bits, block_act_bits, 0.1, "regular4", residual, {})

        self.up5 = FINNUpsamplingBottleneckHAWQ(c4, c5, block_weight_bits["up5"], block_act_bits["up5"], residual=residual)
        self.regular5 = _make_shallow_stage(c5, n5, block_weight_bits, block_act_bits, 0.1, "regular5", residual, {})

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
# Real-weight transfer -- dense_dilation-pattern generalization of
# finn_export_s19_double_mid.py's own transfer_weights() (every stage2/
# stage3 slot here is separable-dilated, no dilation=1 branch needed).
# ---------------------------------------------------------------------------

def _copy_indices(dst_seq, src_seq, indices: list[int], note: str, report: list[str]) -> None:
    n = 0
    for i in indices:
        dst_seq[i].load_state_dict(src_seq[i].state_dict())
        n += sum(p.numel() for p in src_seq[i].parameters())
    report.append(f"  [OK]    {note}: {n} params transferred")


def _fresh(note: str, report: list[str]) -> None:
    report.append(f"  [FRESH] {note}")


def _transfer_regular(dst, src, name: str, report: list[str]) -> None:
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv, src.conv_bn_act, [0, 1], f"{name}.conv<-conv_bn_act (conv+BN)", report)
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}: reduce/conv/expand trailing activations + .act/.requant "
           f"(QuantReLU/QuantDecomposedLeakyAct -> DecomposedLeakyAct, different classes)", report)


def _transfer_sep_dilated(dst, src, name: str, report: list[str]) -> None:
    _copy_indices(dst.reduce, src.reduce, [0, 1], f"{name}.reduce (conv+BN)", report)
    _copy_indices(dst.conv_bn_act[0], src.conv_bn_act[0], [0, 1, 3],
                  f"{name}.conv_bn_act[0] ((k,1)conv + BN + (1,k)conv)", report)
    dst.conv_bn_act[1].load_state_dict(src.conv_bn_act[1].state_dict())
    n_bn = sum(p.numel() for p in src.conv_bn_act[1].parameters())
    report.append(f"  [OK]    {name}.conv_bn_act[1] outer BN: {n_bn} params transferred")
    _copy_indices(dst.expand, src.expand, [0, 1], f"{name}.expand (conv+BN)", report)
    _fresh(f"{name}: inner+outer trailing activations + .act/.requant", report)


def _transfer_initial(dst, src, name: str, report: list[str]) -> None:
    """src (QuantInitialBlock) is conv/pool/bn(out_ch)/act -- structurally
    identical to dst except the single combined BN, which dst splits into
    conv_bn[0:conv_ch]/pool_bn[conv_ch:] (BN is per-channel, so this exact
    index slice is lossless -- not an approximation)."""
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
    _fresh(f"{name}.conv_act/.pool_act/.requant (trailing activations -- same fresh-init "
           f"convention as every other block's trailing activation in this file)", report)


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
           f"main_proj 1x1-conv + parameter-free bilinear interpolate)", report)
    _fresh(f"{name}: reduce/up/expand trailing activations + .main_act/.act/.requant", report)


def transfer_weights(dst: FINNQuantENet26_9_w24HAWQ, src: QuantENet26_9_w24_s14w12_nonneg_block) -> list[str]:
    """Transfers every structurally-identical conv/BN pair from the real,
    PTQ-calibrated-checkpoint-loaded `src` into the FINN-safe `dst`, leaving
    the topology-mismatched components (see module docstring) at their
    fresh-init values."""
    report: list[str] = []

    _transfer_initial(dst.initial, src.initial, "initial", report)

    _transfer_downsampling(dst.down1, src.down1, "down1", report)
    for i, (d, s) in enumerate(zip(dst.regular1, src.regular1)):
        _transfer_regular(d, s, f"regular1.{i}", report)

    _transfer_downsampling(dst.down2, src.down2, "down2", report)

    for stage_name, dst_stage, src_stage in (
        ("stage2", dst.stage2, src.stage2), ("stage3", dst.stage3, src.stage3),
    ):
        for i, (d, s) in enumerate(zip(dst_stage, src_stage)):
            _transfer_sep_dilated(d, s, f"{stage_name}.{i}", report)

    _transfer_upsampling(dst.up4, src.up4, "up4", report)
    for i, (d, s) in enumerate(zip(dst.regular4, src.regular4)):
        _transfer_regular(d, s, f"regular4.{i}", report)

    _transfer_upsampling(dst.up5, src.up5, "up5", report)
    for i, (d, s) in enumerate(zip(dst.regular5, src.regular5)):
        _transfer_regular(d, s, f"regular5.{i}", report)

    dst.final.weight.data.copy_(src.final.weight.data)
    report.append(f"  [OK]    final.weight (conv_transpose kernel): {src.final.weight.numel()} params transferred")
    _fresh("final.bias (real model has a trained bias; FINN-safe final layer requires "
           "bias=False for threshold streamlining -- dropped, not transferable)", report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--block-bits-file", default=str(DEFAULT_BLOCK_BITS_FILE))
    parser.add_argument("--slope-map-file", default=str(DEFAULT_SLOPE_MAP_FILE))
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

    with open(args.slope_map_file) as f:
        leaky_slope_map = json.load(f)

    print("\n=== 1. Building + loading the REAL PTQ-calibrated QuantENet26_9_w24_s14w12_nonneg_block ===")
    real_model = QuantENet26_9_w24_s14w12_nonneg_block(block_weight_bits, block_act_bits, leaky_slope_map=leaky_slope_map)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["network_weights"]
    new_state_dict = {
        (key[7:] if key.startswith("module.") and key not in real_model.state_dict() else key): value
        for key, value in state_dict.items()
    }
    missing, unexpected = real_model.load_state_dict(new_state_dict, strict=True)
    real_model.eval()
    print(f"  Loaded {args.checkpoint} -- missing={missing} unexpected={unexpected}")

    print("\n=== 2. Building the FINN-safe HAWQ mirror + transferring real calibrated weights ===")
    residual = not args.no_residuals
    finn_model = FINNQuantENet26_9_w24HAWQ(
        block_weight_bits, block_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        residual=residual, leaky_slope_map=leaky_slope_map,
    ).eval()

    report = transfer_weights(finn_model, real_model)
    print("\n".join(report))
    n_ok = sum(1 for line in report if "[OK]" in line)
    n_fresh = sum(1 for line in report if "[FRESH]" in line)
    print(f"\n  {n_ok} components transferred from the real calibrated checkpoint, {n_fresh} components fresh-initialized.")

    print("\n=== 3. Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = finn_model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    suffix = "_no_res" if not residual else ""
    name = f"quantEnet_26_9_w24_hawq_joint_ptq_int8{suffix}"
    export_model(finn_model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
