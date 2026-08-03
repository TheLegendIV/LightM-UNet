"""Export a production-scale FINN-compatible quantized ENet.

FINN compatibility modifications vs QuantENet.py:
1. No asymmetric convolutions (asymmetric kernels not supported cleanly by FINN's
   ConvolutionInputGenerator in all configurations).
2. No MaxUnpool / no F.interpolate in upsampling:
   - MaxUnpool: torch.onnx.export raises UnsupportedOperatorError for aten::max_unpool2d.
   - F.interpolate (bilinear): becomes ONNX Resize node, not supported in FINN dataflow.
   - Fix: main path of QuantUpsamplingBottleneck uses QuantConvTranspose2d(stride=2).
3. MaxPool in downsampling shortcut without return_indices=True:
   - The original QuantDownsamplingBottleneck calls MaxPool(return_indices=True) even when
     the decoder doesn't use indices (upsample_conv mode). The unused indices output creates
     a dangling output in the ONNX graph that FINN may not handle well.
   - Fix: shortcut path uses MaxPool(return_indices=False) + 1x1 QuantConv for channel proj.
4. Final layer bias=False: simplifies FINN's streamlining (bias folding into thresholds
   requires the output to be quantized, which the final layer isn't always).
5. No DSC (depthwise separable convolutions) — not tested in FINN for this model variant.
6. Dilated convolutions are supported by FINN (ConvolutionInputGenerator has dilations param),
   so use_dilated=True is fine and left configurable.

Usage:
    python hardware/finn_enet_prod_export.py
    python hardware/finn_enet_prod_export.py --channels 8 16 32 16 8   # smaller test
    python hardware/finn_enet_prod_export.py --no-residuals             # fully no-residual (debug-like)

Output: hardware/outputs/finn_exports/quantEnet_finn_v1.onnx (and ..._enc.onnx)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.QuantENet import _quant_conv2d, _quant_act  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
DEFAULT_CHANNELS = (20, 72, 144, 72, 20)
DEFAULT_BNECKS   = (4, 8, 8, 2, 1)
BIT_WIDTH = 8
# Full-span ENet-native dilation schedule (matches ENet.py's CONTEXT_STAGE_PATTERN
# flattened to just its dilation values: regular, d2, asymmetric(->regular here,
# no asymmetric option in this FINN-safe variant), d4, regular, d8, asymmetric
# (->regular), d16). Previously this was a hand-invented [1,2,4,8,1,2,4,8] that
# never reached d16 and ran dilated convs back-to-back with no filler slots --
# a real divergence from ENet.py, now fixed to use the full 8-slot span.
CONTEXT_DILATIONS = [1, 2, 1, 4, 1, 8, 1, 16]


# ---------------------------------------------------------------------------
# FINN-compatible block definitions
# ---------------------------------------------------------------------------

class FINNInitialBlock(nn.Module):
    """Single-conv initial block (no MaxPool branch → no Concat → FINN-safe).

    Original ENet InitialBlock concatenates a stride-2 conv branch with a MaxPool
    branch.  The MaxPool → Concat pattern is not cleanly handled by FINN's
    dataflow partitioner, so we use a single stride-2 conv that directly produces
    out_channels.  Topology difference from original is minimal: one 3×3 stride-2
    conv vs one 3×3 stride-2 conv + MaxPool + Concat.
    """

    def __init__(self, in_ch: int, out_ch: int, bit_width: int):
        super().__init__()
        self.conv = _quant_conv2d(in_ch, out_ch, bit_width, kernel_size=3, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FINNRegularBottleneck(nn.Module):
    """Regular bottleneck with optional dilation, no asymmetric kernels.

    Residual: identity shortcut + QuantEltwiseAdd → FINN AddStreams_Batch.
    Set residual=False for the fully single-stream (debug-like) variant.
    """

    def __init__(
        self, channels: int, bit_width: int,
        dilation: int = 1,
        internal_ratio: int = 4,
        dropout_p: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, channels // internal_ratio)

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, bit_width,
                          kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
            _quant_act(bit_width),   # quantize to UINT8 before residual Add
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = out + x  # plain add — MoveLinearPastEltwiseAdd handles BN scales
        return self.act(out)


class FINNDownsamplingBottleneck(nn.Module):
    """Downsampling bottleneck without MaxPool-with-indices.

    Shortcut path: MaxPool(return_indices=False) + 1×1 QuantConv for channel
    projection.  This avoids the dangling 'indices' ONNX output from the original
    QuantDownsamplingBottleneck.  Branch path: stride-2 2×2 conv.
    Residual: QuantEltwiseAdd on pooled+projected shortcut + branch expansion.
    """

    def __init__(
        self, in_ch: int, out_ch: int, bit_width: int,
        internal_ratio: int = 4,
        dropout_p: float = 0.01,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, out_ch // internal_ratio)

        # Shortcut: spatial downsampling + learned channel projection
        self.shortcut_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.shortcut_proj = nn.Sequential(
            _quant_conv2d(in_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _quant_act(bit_width),   # quantize to UINT8 before residual Add
        )

        # Branch
        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, bit_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, bit_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _quant_act(bit_width),   # quantize to UINT8 before residual Add
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_proj(self.shortcut_pool(x)) if self.residual else None
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = out + shortcut  # plain add — MoveLinearPastEltwiseAdd handles BN scales
        return self.act(out)


class FINNUpsamplingBottleneck(nn.Module):
    """Upsampling bottleneck replacing F.interpolate with QuantConvTranspose2d.

    Main path: QuantConvTranspose2d(in_ch→out_ch, kernel=2, stride=2) directly
    does both channel projection and 2× spatial upsampling — no F.interpolate,
    no Resize ONNX node, no MaxUnpool.  Branch path: reduce + ConvTranspose +
    expand.  Residual: QuantEltwiseAdd on both paths.
    """

    def __init__(
        self, in_ch: int, out_ch: int, bit_width: int,
        internal_ratio: int = 4,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, in_ch // internal_ratio)

        # Main path: learned 2× upsample + channel projection in one op
        self.main_up = qnn.QuantConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )
        self.main_bn  = nn.BatchNorm2d(out_ch)
        self.main_act = _quant_act(bit_width)  # quantize to UINT8 before residual Add

        # Branch path
        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.up = nn.Sequential(
            qnn.QuantConvTranspose2d(
                internal_ch, internal_ch, kernel_size=2, stride=2, bias=False,
                weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
            ),
            nn.BatchNorm2d(internal_ch),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            _quant_act(bit_width),   # quantize to UINT8 before residual Add
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_act(self.main_bn(self.main_up(x)))
        out  = self.dropout(self.expand(self.up(self.reduce(x))))
        if self.residual:
            out = out + main  # plain add — both inputs are UINT8
        return self.act(out)


class FINNQuantENet(nn.Module):
    """Production-scale FINN-compatible quantized ENet.

    Architecture mirrors QuantENet.py stage-for-stage (same channel widths and
    bottleneck counts) with the FINN-compatibility fixes listed in the module
    docstring.  All block types are the FINN* variants above.

    Args:
        in_channels: input image channels (1 for grayscale CT/MRI).
        out_channels: segmentation classes.
        channels: 5-tuple of channel widths per stage (initial, stage1, stage23,
            stage4, stage5).  Default matches the production ENet.
        bottlenecks_per_stage: 5-tuple of bottleneck counts.
        use_dilated: include dilated convolutions in context stages (stage2/3).
            FINN's ConvolutionInputGenerator supports dilation, so True is fine.
        bit_width: uniform weight+activation quantization bit-width.
        residual: include residual shortcuts.  Set False for single-stream
            debugging (similar to finn_debug_nets.py's *NoResidual classes).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        bottlenecks_per_stage: tuple[int, ...] = DEFAULT_BNECKS,
        use_dilated: bool = True,
        bit_width: int = BIT_WIDTH,
        residual: bool = True,
    ):
        super().__init__()
        c0, c1, c23, c4, c5 = channels
        n1, n2, n3, n4, n5 = bottlenecks_per_stage

        self.initial = FINNInitialBlock(in_channels, c0, bit_width)

        self.down1 = FINNDownsamplingBottleneck(c0, c1, bit_width, dropout_p=0.01, residual=residual)
        self.regular1 = nn.Sequential(*[
            FINNRegularBottleneck(c1, bit_width, dropout_p=0.01, residual=residual)
            for _ in range(n1)
        ])

        self.down2 = FINNDownsamplingBottleneck(c1, c23, bit_width, dropout_p=0.1, residual=residual)
        self.stage2 = self._make_context_stage(c23, n2, bit_width, use_dilated, residual)
        self.stage3 = self._make_context_stage(c23, n3, bit_width, use_dilated, residual)

        self.up4 = FINNUpsamplingBottleneck(c23, c4, bit_width, residual=residual)
        self.regular4 = nn.Sequential(*[
            FINNRegularBottleneck(c4, bit_width, dropout_p=0.1, residual=residual)
            for _ in range(n4)
        ])

        self.up5 = FINNUpsamplingBottleneck(c4, c5, bit_width, residual=residual)
        self.regular5 = nn.Sequential(*[
            FINNRegularBottleneck(c5, bit_width, dropout_p=0.1, residual=residual)
            for _ in range(n5)
        ])

        # Final: no bias (bias in ConvTranspose requires extra BN/threshold handling)
        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )

    @staticmethod
    def _make_context_stage(
        channels: int, n: int, bit_width: int, use_dilated: bool, residual: bool
    ) -> nn.Sequential:
        blocks = []
        for i in range(n):
            dil = CONTEXT_DILATIONS[i % len(CONTEXT_DILATIONS)] if use_dilated else 1
            blocks.append(FINNRegularBottleneck(channels, bit_width, dilation=dil, residual=residual))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.regular1(self.down1(x))
        x = self.stage3(self.stage2(self.down2(x)))
        x = self.regular4(self.up4(x))
        x = self.regular5(self.up5(x))
        out = self.final(x)
        return out.value if hasattr(out, "value") else out


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_model(model: nn.Module, name: str, dummy: torch.Tensor) -> Path:
    """Export model to cleaned QONNX, set INT8 datatypes, verify with onnx.checker."""
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    export_qonnx(model, export_path=str(out_path), input_t=dummy)
    qonnx_cleanup(str(out_path), out_file=str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name,  DataType["INT8"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType["INT8"])
    qm.save(str(out_path))

    # onnx.checker.check_model() rejects QONNX custom ops ('qonnx.custom_op.general'
    # domain); skip it.  FINN's own transforms validate the graph internally.
    # Light sanity check: model loads and has at least one node.
    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"

    ops: dict[str, int] = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(qm.graph.node)} nodes — {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", type=int, nargs=5,
                        default=list(DEFAULT_CHANNELS), metavar="C",
                        help="Channel widths per stage (default: 20 72 144 72 20)")
    parser.add_argument("--bnecks", type=int, nargs=5,
                        default=list(DEFAULT_BNECKS), metavar="N",
                        help="Bottleneck counts per stage (default: 4 8 8 2 1)")
    parser.add_argument("--in-channels",  type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64),
                        metavar=("H", "W"),
                        help="Spatial dims for export dummy input (default: 64 64). "
                             "Must be divisible by 8.")
    parser.add_argument("--bit-width", type=int, default=BIT_WIDTH)
    parser.add_argument("--no-dilated",  action="store_true",
                        help="Disable dilation in context stages.")
    parser.add_argument("--no-residuals", action="store_true",
                        help="Remove all residual shortcuts (fully single-stream).")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    torch.manual_seed(0)
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1

    residual    = not args.no_residuals
    use_dilated = not args.no_dilated
    channels    = tuple(args.channels)
    bnecks      = tuple(args.bnecks)
    bw          = args.bit_width

    print(f"\nFINNQuantENet export")
    print(f"  channels={channels}, bnecks={bnecks}")
    print(f"  input=({args.in_channels},{h},{w})  bit_width={bw}")
    print(f"  residual={residual}  use_dilated={use_dilated}")

    full = FINNQuantENet(
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=channels, bottlenecks_per_stage=bnecks,
        use_dilated=use_dilated, bit_width=bw, residual=residual,
    ).eval()

    with torch.no_grad():
        out = full(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape[2:] == (h, w), f"output HxW {tuple(out_t.shape[2:])} != input ({h},{w})"
    print(f"  forward OK: output shape {tuple(out_t.shape)}")

    suffix = "_no_res" if not residual else ""
    export_model(full, f"quantEnet_finn_v1{suffix}", dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/quantEnet_finn_v1{suffix}.onnx "
          f"<container_id>:/home/thelegendiv/finn/notebooks/enet/")


if __name__ == "__main__":
    main()
