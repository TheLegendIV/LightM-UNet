"""Export a FINN-compatible, PER-BLOCK-heterogeneous-bit-width mirror of
S19 (19_reginterleaved_separable_nonneg_block_double_mid), using
compression/hawq/block_bits_s19.json's per-block weight/act bit-width
choice (2/4/8 bits, one independent choice per BLOCK_NAMES entry -- see
enet/nnunetv2/nets/QuantENetS19Block.py, the real-model analogue this
mirrors) instead of a single uniform bit-width across the whole network.

No trained checkpoint exists for this exact per-block bit-width config, so
(same established convention as finn_export_s19_single_block.py/
finn_export_s13_leaky_frozen.py/every other resource-probe export in this
repo) this uses FRESH (torch.manual_seed(0)) conv/BN weights throughout --
FINN's LUT/BRAM/DSP resource estimate and OOC synthesis result depend only
on architecture + bit-width + which nodes exist, not on weight values. The
one place a REAL (not fresh) value is used is the per-block LeakyReLU
negative_slope, taken from the real S19 double_mid slope map (same
DEFAULT_SLOPE_MAP_FILE finn_export_s19_double_mid.py uses) -- this keeps
the comparison against the existing s19_double_mid_partition_2_ooc_synth
baseline apples-to-apples (same activation function/graph topology, only
the per-block bit-width and folding differ).

Reuses every generic FINN-safe helper from finn_export_s13_leaky_frozen.py
(DecomposedLeakyAct, _make_act_factory, _plain_relu_factory, _val,
_requant_factory, export_model) exactly like finn_export_s19_double_mid.py/
finn_export_s19_single_block.py already do. The block classes themselves
(FINNInitialBlockHAWQ/FINNDownsamplingBottleneckHAWQ/
FINNUpsamplingBottleneckHAWQ/FINNRegularBottleneckHAWQ/
FINNRegularBottleneckSepDilatedHAWQ) are new -- structurally identical to
their non-HAWQ counterparts in finn_export_s13_leaky_frozen.py, just
parameterized by TWO bit-widths (weight_bits for every _quant_conv2d call,
act_bits for every act_factory call) instead of one shared `bit_width`,
since block_bits_s19.json's weight/act values genuinely differ per block
(e.g. stage2.2: weight=2, act=4).

Usage (run inside the pytorch training container):
    python hardware/finn_export_s19_hawq_block.py

Output: hardware/outputs/finn_exports/quantEnet_s19_hawq_block_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_s19_hawq_block_int8.onnx \\
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
from nnunetv2.nets.ENet import DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN  # noqa: E402
from nnunetv2.nets.QuantENetS19Block import BLOCK_NAMES  # noqa: E402
from finn_export_s13_leaky_frozen import (  # noqa: E402
    _make_act_factory, _plain_relu_factory, _requant_factory, _val, export_model,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
CHANNELS = (4, 16, 32, 16, 4)          # initial, s1, s23 (shared), s4, s5 -- S19's own
BOTTLENECKS_PER_STAGE = (4, 12, 12, 2, 1)
DEFAULT_SLOPE_MAP_FILE = (
    REPO_ROOT / "compression" / "post-quantization" / "slope_maps"
    / "19_reginterleaved_separable_nonneg_block_double_mid.json"
)
DEFAULT_BLOCK_BITS_FILE = REPO_ROOT / "compression" / "hawq" / "block_bits_s19.json"


# ---------------------------------------------------------------------------
# Per-block-bit-width FINN-safe block definitions (weight_bits/act_bits may
# differ; every other FINN*Bottleneck class in this repo hardcodes one
# shared `bit_width` for both).
# ---------------------------------------------------------------------------

class FINNInitialBlockHAWQ(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weight_bits: int, act_bits: int, act_factory):
        super().__init__()
        self.conv = _quant_conv2d(in_ch, out_ch, weight_bits, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_factory(out_ch, act_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


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
    """Decoder-half block: always plain QuantReLU (never leaky), matching
    ENet.py/QuantENet.py hardcoding relu=True on up4/up5/regular4/regular5."""

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
    factoring -- every stage2/stage3 dilated slot."""

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
# Model assembly -- same topology as FINNQuantENetS19DoubleMid, just every
# block looks up its own (weight_bits, act_bits) pair by BLOCK_NAMES key.
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
    pattern = DENSE_DILATION_REG_INTERLEAVED_DOUBLE_MID_PATTERN
    blocks = []
    for i in range(n):
        slot = dict(pattern[i % len(pattern)])
        slot.pop("reg_bottleneck", False)
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


class FINNQuantENetS19BlockHAWQ(nn.Module):
    """FINN-compatible, per-BLOCK-bit-width mirror of QuantENetS19Block --
    same architecture as FINNQuantENetS19DoubleMid, but block_weight_bits/
    block_act_bits (one entry per BLOCK_NAMES) replace the single shared
    `bit_width`. No defaulting for missing keys, same philosophy
    QuantENetS19Block.py already uses."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    print("\n=== Building fresh-weight, per-block-bit-width FINN-safe S19 (HAWQ block config) ===")
    torch.manual_seed(0)
    residual = not args.no_residuals
    model = FINNQuantENetS19BlockHAWQ(
        block_weight_bits, block_act_bits,
        in_channels=args.in_channels, out_channels=args.out_channels,
        residual=residual, leaky_slope_map=leaky_slope_map,
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
    name = f"quantEnet_s19_hawq_block_int8{suffix}"
    export_model(model, name, dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/{name}.onnx <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
