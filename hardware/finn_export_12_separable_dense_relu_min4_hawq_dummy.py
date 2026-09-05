"""Export a FINN-compatible, PER-BLOCK-heterogeneous-bit-width mirror of
nnUNetTrainerENet_12_separable_dense_relu ("S12.1" -- S5.6's own
separable_dilated+dense_dilation recipe at native S8.2/S5.6-family width,
CHANNELS=(4,16,32,16,4), but with plain ReLU (use_prelu=0) instead of
per-channel PReLU -- see compression/hawq/config_12_separable_dense_relu.py),
using the per-block HAWQ scheme in
compression/hawq/block_bits_12_separable_dense_relu_min4.json (one
independent (weight_bits, act_bits) choice per block, min-4-bit floor).

No trained checkpoint for this exact per-block bit-width config is wired up
here, so (same established convention as finn_export_26_5_w24_hawq_joint.py/
finn_export_s19_hawq_block.py) this uses FRESH (torch.manual_seed(0))
conv/BN weights throughout -- FINN's LUT/BRAM/DSP resource estimate and OOC
synthesis result depend only on architecture + bit-width + which nodes
exist, not on weight values.

USE_PRELU=False for this config collapses EVERY activation (encoder and
decoder alike) to plain ReLU -- so unlike the 26_5_w24/26_9_w24/s19-family
HAWQ exports (real or post-hoc-averaged per-channel PReLU slopes, decomposed
via DecomposedLeakyAct), there is no slope map at all here: every activation
site uses finn_export_s13_leaky_frozen.py's own _plain_relu_factory
(itself a signed-Int8-consistent epsilon-slope DecomposedLeakyAct, the same
"always ReLU" convention every decoder half of every sibling HAWQ export in
this repo already uses -- see that factory's own docstring for why a literal
ONNX Relu node isn't used instead).

Uses the REAL Concat-based FINNInitialBlockHAWQ initial block (learned conv
branch + parameter-free MaxPool branch merged via FINN's StreamingConcat),
same as finn_export_26_5_w24_hawq_joint.py/finn_export_26_9_w24_hawq_joint_ptq.py
-- NOT the older single-full-out_channels-conv simplification some earlier
export scripts in this repo still use.

Block classes (FINNInitialBlockHAWQ/FINNDownsamplingBottleneckHAWQ/
FINNUpsamplingBottleneckHAWQ/FINNRegularBottleneckHAWQ/
FINNRegularBottleneckSepDilatedHAWQ) are structurally identical to
finn_export_26_5_w24_hawq_joint.py's own, just with act_factory hardcoded to
_plain_relu_factory everywhere (no per-block slope to look up).

Reuses _plain_relu_factory/_requant_factory/_val/export_model from
finn_export_s13_leaky_frozen.py (copied there, not imported here directly --
same "don't import a __main__-driven sibling script" convention every export
script in this repo already follows... except this one DOES import from
s13, since s13's own helpers are pure functions with no argparse/CLI
side effects at import time, same as every other *_hawq_joint.py script).

IMPORTANT (build-side, not this script): DecomposedLeakyAct's pre/out
quantizers are SIGNED Int8 -- on a randomly-initialized/untrained network
this is known to trip a degenerate out_bias=0 calibration bug in qonnx's
Quant->MultiThreshold conversion (see hardware/finn_enet_build_decomposed_
prelu.py's docstring). The FINN build for this export MUST insert
finn_enet_ip_build_partitioned_8way.py's own _fixup_degenerate_signed_bias
step between step_enet_streamline and step_enet_convert_to_hw (already part
of enet_ip_partitioned_8way_steps, no extra action needed as long as this
export is built through that same step list).

Usage (run inside the pytorch training container, e.g. `lightm_pytorch`):
    docker exec lightm_pytorch python /workspace/LightM-UNet/hardware/finn_export_12_separable_dense_relu_min4_hawq_dummy.py

Output: hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_min4_hawq_dummy_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_12_separable_dense_relu_min4_hawq_dummy_int8.onnx \\
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
from finn_export_s13_leaky_frozen import _plain_relu_factory, _requant_factory, _val, export_model  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)          # initial, s1, s23 (shared), s4, s5 -- 12_separable_dense_relu's own
BOTTLENECKS_PER_STAGE = (4, 8, 8, 2, 1)
DEFAULT_BLOCK_BITS_FILE = REPO_ROOT / "compression" / "hawq" / "block_bits_12_separable_dense_relu_min4.json"


# ---------------------------------------------------------------------------
# Per-block-bit-width FINN-safe block definitions -- structurally identical
# to finn_export_26_5_w24_hawq_joint.py's own, minus the act_factory/slope
# threading (every site here is plain ReLU, see module docstring).
# ---------------------------------------------------------------------------

class FINNInitialBlockHAWQ(nn.Module):
    """Real Concat-based initial block: learned conv branch (in_ch ->
    out_ch-in_ch) and parameter-free MaxPool branch (in_ch -> in_ch) merged
    via FINN's StreamingConcat -- see finn_export_26_5_w24_hawq_joint.py's
    identical class for the full rationale."""

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
    """Decoder-half block: always plain ReLU (same convention as every
    other FINN*HAWQ export in this repo)."""

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
    """Dilated context-stage bottleneck, (k,1)+(1,k) DENSE (non-depthwise)
    separable-dilated factoring -- every stage2/stage3 slot
    (DENSE_DILATION_PATTERN has no plain, dilation=1 slots at all)."""

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


# ---------------------------------------------------------------------------
# Model assembly -- same topology as QuantENet's 12_separable_dense_relu
# recipe, every block looks up its own (weight_bits, act_bits) pair by
# block name.
# ---------------------------------------------------------------------------

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
    """FINN-compatible, per-BLOCK-bit-width mirror of the (fictional,
    never-quantized-in-this-repo) QuantENet12_separable_dense_relu recipe --
    block_weight_bits/block_act_bits (one entry per compression/hawq/
    block_bits_12_separable_dense_relu_min4.json key) replace the single
    shared bit_width. Every activation is plain ReLU (USE_PRELU=False)."""

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

    print("\n=== Building fresh-weight, per-block-bit-width FINN-safe 12_separable_dense_relu (HAWQ min4 config) ===")
    torch.manual_seed(0)
    residual = not args.no_residuals
    model = FINNQuantENet12SepDenseReluHAWQ(
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
    name = f"quantEnet_12_separable_dense_relu_min4_hawq_dummy_int8{suffix}"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
