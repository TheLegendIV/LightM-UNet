"""Export a FINN-compatible, PER-BLOCK-heterogeneous-bit-width mirror of
nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_w16_perblock (DSC-no-projection,
dense_dilation, plain ReLU everywhere -- USE_PRELU=False), using
compression/hawq/block_bits_8_2_relu_no_reg_w16_acc2x_min4_joint.json's
per-block JOINT (weight_bits, act_bits) choice.

Architecture (see enet/nnunetv2/training/nnUNetTrainer/
nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_w16_perblock.py):
    CHANNELS = (4, 8, 16, 8, 4), BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
    CONTEXT_PATTERN = dense_dilation (DENSE_DILATION_PATTERN, 8 slots,
        dilations [2,4,8,16,2,4,8,16], applied identically to stage2/stage3)
    USE_ASYMMETRIC=False, SEPARABLE_DILATED=False, USE_PRELU=False,
    USE_DSC=False, DSC_NO_PROJECTION=True (unscoped/"all"). Architecturally
    IDENTICAL to nnUNetTrainerENet_8_2_relu_no_reg_w20
    (finn_export_8_2_relu_no_reg_w20_hawq.py) except channel widths -- every
    FINN-safe block class below is a byte-for-byte copy of that file's own.

The REAL trained checkpoint for this config
(data/nnUNet_results/Dataset509_ARCADE_1x1_4c/nnUNetTrainerCombinedQuantENet_
8_2_relu_no_reg_w16_perblock__nnUNetPlans__2d/fold_0/) has not synced to this
workspace yet (training logs present, no .pth file) -- per this session's
explicit "just use dummy weights for now" instruction, this uses FRESH
(torch.manual_seed(0)) conv/BN weights throughout, no checkpoint loading.
Same established convention as every other resource-probe export in this
repo (finn_export_8_2_relu_no_reg_w20_hawq.py, finn_export_s19_hawq_block.py):
FINN's LUT/BRAM/DSP resource estimate and OOC synthesis result depend only
on architecture + bit-width + which nodes exist, not on weight values. A
real-weight version (finn_export_8_2_relu_no_reg_w16_acc2x_hawq_real.py,
with transfer_weights() from the real CombinedQuantENet checkpoint) already
exists for once the checkpoint syncs.

Reuses every generic FINN-safe helper from finn_export_s13_leaky_frozen.py
(_plain_relu_factory, _val, _requant_factory, export_model).

Usage (run inside the pytorch training container):
    python hardware/finn_export_8_2_relu_no_reg_w16_acc2x_hawq_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_8_2_relu_no_reg_w16_acc2x_hawq_dummy.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_8_2_relu_no_reg_w16_acc2x_hawq_dummy.onnx \\
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
from finn_export_s13_leaky_frozen import _plain_relu_factory, _requant_factory, _val, export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 8, 16, 8, 4)            # initial, s1, s23 (shared), s4, s5 -- w16's own (narrower than w20's (4,10,20,10,4))
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DENSE_DILATIONS = [2, 4, 8, 16, 2, 4, 8, 16]
DEFAULT_BLOCK_BITS_FILE = (
    REPO_ROOT / "compression" / "hawq" / "block_bits_8_2_relu_no_reg_w16_acc2x_min4_joint.json"
)
BLOCK_NAMES = (
    ["initial", "down1"] + [f"regular1.{i}" for i in range(4)]
    + ["down2"] + [f"stage2.{i}" for i in range(8)] + [f"stage3.{i}" for i in range(8)]
    + ["up4"] + [f"regular4.{i}" for i in range(2)]
    + ["up5"] + [f"regular5.{i}" for i in range(1)]
    + ["final"]
)


# ---------------------------------------------------------------------------
# Per-block-bit-width FINN-safe block definitions. Every activation site is
# plain ReLU (_plain_relu_factory) -- this config has USE_PRELU=False, no
# slope map/leaky decomposition needed anywhere (encoder or decoder).
# ---------------------------------------------------------------------------

class FINNInitialBlockHAWQ(nn.Module):
    """Single stride-2 conv producing the full channel count directly (no
    MaxPool-branch concat) -- same topology fix every export in this repo
    uses (see hardware/README.md)."""

    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int):
        super().__init__()
        self.conv = _quant_conv2d(in_ch, out_ch, weight_bits, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _plain_relu_factory(out_ch, act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FINNDownsamplingBottleneckHAWQ(nn.Module):
    """Channel-changing block -- keeps the normal reduce/conv/expand +
    MaxPool-shortcut-projection structure (DSC-no-projection cannot apply
    where channels change)."""

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
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_proj(self.shortcut_pool(x)) if self.residual else None
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(shortcut))
        return self.act(out)


class FINNUpsamplingBottleneckHAWQ(nn.Module):
    """Channel-changing block -- keeps the normal reduce/up/expand +
    ConvTranspose-main-path structure (DSC-no-projection cannot apply where
    channels change)."""

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


class FINNDSCNoProjectionBottleneckHAWQ(nn.Module):
    """Depthwise KxK (groups=channels, dilation-aware) + pointwise 1x1, no
    reduce/expand projection. Adds a requant-before-residual-Add since
    neighbouring blocks' bit-widths genuinely differ."""

    def __init__(self, channels: int, weight_bits: int, act_bits: int,
                 dilation: int = 1, dropout_p: float = 0.1, residual: bool = True):
        super().__init__()
        self.residual = residual
        padding = dilation
        self.conv = nn.Sequential(
            _quant_conv2d(channels, channels, weight_bits, kernel_size=3, padding=padding,
                          dilation=dilation, groups=channels),
            nn.BatchNorm2d(channels),
            _plain_relu_factory(channels, act_bits),
            _quant_conv2d(channels, channels, weight_bits, kernel_size=1),
            nn.BatchNorm2d(channels),
            _plain_relu_factory(channels, act_bits),  # quantize before residual Add
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = _plain_relu_factory(channels, act_bits)
        self.requant = _requant_factory(act_bits) if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv(x))
        if self.residual:
            out = self.requant(_val(out)) + self.requant(_val(x))
        return self.act(out)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

class FINNQuantENet8w16HAWQ(nn.Module):
    """FINN-compatible, per-BLOCK-bit-width mirror of
    nnUNetTrainerCombinedQuantENet_8_2_relu_no_reg_w16_perblock. No
    defaulting for missing keys -- same philosophy every other *HAWQ model
    class in this repo uses."""

    def __init__(
        self, block_weight_bits: dict[str, int], block_act_bits: dict[str, int],
        in_channels: int = 1, out_channels: int = 5,
        channels: tuple[int, ...] = CHANNELS, bottlenecks_per_stage: tuple[int, ...] = BOTTLENECKS_PER_STAGE,
        residual: bool = True,
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

        self.initial = FINNInitialBlockHAWQ(in_channels, c0, block_weight_bits["initial"], block_act_bits["initial"])

        self.down1 = FINNDownsamplingBottleneckHAWQ(
            c0, c1, block_weight_bits["down1"], block_act_bits["down1"], dropout_p=0.01, residual=residual,
        )
        self.regular1 = nn.Sequential(*[
            FINNDSCNoProjectionBottleneckHAWQ(
                c1, block_weight_bits[f"regular1.{i}"], block_act_bits[f"regular1.{i}"],
                dilation=1, dropout_p=0.01, residual=residual,
            )
            for i in range(n1)
        ])

        self.down2 = FINNDownsamplingBottleneckHAWQ(
            c1, c23, block_weight_bits["down2"], block_act_bits["down2"], dropout_p=0.1, residual=residual,
        )
        self.stage2 = self._make_dense_context_stage(c23, n2, block_weight_bits, block_act_bits, "stage2", residual)
        self.stage3 = self._make_dense_context_stage(c23, n3, block_weight_bits, block_act_bits, "stage3", residual)

        self.up4 = FINNUpsamplingBottleneckHAWQ(c23, c4, block_weight_bits["up4"], block_act_bits["up4"], residual=residual)
        self.regular4 = nn.Sequential(*[
            FINNDSCNoProjectionBottleneckHAWQ(
                c4, block_weight_bits[f"regular4.{i}"], block_act_bits[f"regular4.{i}"],
                dilation=1, dropout_p=0.1, residual=residual,
            )
            for i in range(n4)
        ])
        self.up5 = FINNUpsamplingBottleneckHAWQ(c4, c5, block_weight_bits["up5"], block_act_bits["up5"], residual=residual)
        self.regular5 = nn.Sequential(*[
            FINNDSCNoProjectionBottleneckHAWQ(
                c5, block_weight_bits[f"regular5.{i}"], block_act_bits[f"regular5.{i}"],
                dilation=1, dropout_p=0.1, residual=residual,
            )
            for i in range(n5)
        ])

        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=False,
            weight_bit_width=block_weight_bits["final"], weight_quant=Int8WeightPerTensorFloat,
        )

    @staticmethod
    def _make_dense_context_stage(channels: int, n: int, block_weight_bits: dict, block_act_bits: dict,
                                   name_prefix: str, residual: bool) -> nn.Sequential:
        blocks = []
        for i in range(n):
            dilation = DENSE_DILATIONS[i % len(DENSE_DILATIONS)]
            name = f"{name_prefix}.{i}"
            blocks.append(FINNDSCNoProjectionBottleneckHAWQ(
                channels, block_weight_bits[name], block_act_bits[name],
                dilation=dilation, dropout_p=0.1, residual=residual,
            ))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.regular1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.stage3(x)
        x = self.regular4(self.up4(x))
        x = self.regular5(self.up5(x))
        out = self.final(x)
        return out.value if hasattr(out, "value") else out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    print("\n=== Building fresh-weight, per-block-bit-width FINN-safe 8_2_relu_no_reg_w16 (HAWQ acc2x_min4 joint) ===")
    torch.manual_seed(0)
    residual = not args.no_residuals
    model = FINNQuantENet8w16HAWQ(
        block_weight_bits, block_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        residual=residual,
    ).eval()
    print(f"  weight_bits per block: {block_weight_bits}")
    print(f"  act_bits per block: {block_act_bits}")

    print("\n=== Forward-pass sanity check + QONNX export ===")
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[2:] == (h, w), f"output HxW {tuple(out.shape[2:])} != input ({h},{w})"
    assert out.shape[1] == args.out_channels, f"output channels {out.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out.shape)}")

    suffix = "_no_res" if not residual else ""
    name = f"quantEnet_8_2_relu_no_reg_w16_acc2x_hawq_dummy{suffix}"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
