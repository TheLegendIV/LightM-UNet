"""Bare-minimum-structure QuantENet nets, for debugging FINN front-end
QONNX streamlining -- NOT for training or accuracy. Random (untrained)
weights only.

Motivation: the FINN streamlining transform sequence has been failing on
the full-size QuantENet export, and the full graph is too large to debug
transform-by-transform. These nets keep every *unique* op type QuantENet's
streamlining needs to handle (initial block's concat, downsampling
bottleneck's zero-pad + residual add, a context-stage regular bottleneck,
upsampling bottleneck's interpolate + residual add, the final transposed
conv) while cutting every dimension that's just repetition of an op FINN
already has to handle elsewhere:
  - 1 bottleneck per stage (bottlenecks_per_stage patterns only vary
    *which* bottleneck runs, not whether streamlining handles it).
  - Uniform smallest channel width (4) at every stage, including the
    initial block (4 > in_channels=1, the only constraint QuantInitialBlock
    enforces).
  - stage3 dropped entirely: stage2 and stage3 are both
    `_make_context_stage` calls over the exact same CONTEXT_STAGE_PATTERN,
    i.e. structurally identical from FINN's perspective -- keeping one
    covers the op, keeping both would just double the transform's runtime
    on identical nodes.
  - decoder_type='upsample_conv' (bilinear interpolate), not max_unpool --
    see QuantENetDebugFull's docstring: max_unpool doesn't even reach QONNX,
    PyTorch's ONNX exporter has no symbolic for aten::max_unpool2d at all.

Two variants, matching the two things worth isolating in FINN debugging --
does streamlining break in the encoder alone, or only once the decoder's
max-unpool/transposed-conv ops are added:
  - QuantENetDebugEncoder: initial -> down1 -> regular1 -> down2 -> stage2.
  - QuantENetDebugFull: the above -> up4 -> regular4 -> up5 -> regular5 ->
    final transposed conv.

Both bound their input AND output to [-1, 1] (FINN-friendly signed range,
matching the cybersec-mlp QONNX example's bipolar-input pattern) via a
fixed-range QuantHardTanh rather than the runtime-calibrated
Int8ActPerTensorFloat activation quantizers used elsewhere in QuantENet --
with random weights and no calibration data, a stats-derived scale would
be arbitrary, whereas HardTanh(-1, 1) gives a fixed, known scale
regardless of what the random weights produce. This is IN ADDITION to
QuantInitialBlock's own internal input quantizer (a second, redundant
quant node FINN's streamlining has to fold) -- deliberately realistic
rather than special-cased away, since real QONNX exports have the same
redundancy.

Quantization: int8 weights (Int8WeightPerTensorFloat) and int8 signed
activations throughout, EXCEPT QuantReLU outputs, which are uint8
(Uint8ActPerTensorFloat) -- same convention as QuantENet.py's
`_quant_conv2d`/`_quant_act` (reused directly, not reimplemented, so this
file can't drift from QuantENet's own quantization scheme).

Usage:
    python compression/finn_debug_nets.py
    python compression/finn_debug_nets.py --input-hw 32 32 --channels 4

Output: compression/configs/finn_debug/quantEnet_debug_{enc,full}.onnx,
cleaned QONNX, plus the FINN-ONNX conversion (ConvertQONNXtoFINN) IF `finn`
is importable in this environment -- if not (as in this repo's normal
training container, which only has qonnx/brevitas, not FINN itself), this
prints what's missing and stops after the cleaned-QONNX file, same
graceful-degradation pattern as finn_resource_probe.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "enet"))
from nnunetv2.nets.QuantENet import (  # noqa: E402
    QuantInitialBlock,
    QuantDownsamplingBottleneck,
    QuantRegularBottleneck,
    QuantUpsamplingBottleneck,
)

OUT_DIR = Path(__file__).resolve().parent / "configs" / "finn_debug"
DEBUG_CHANNELS = 4  # smallest usable width: max(1, 4 // internal_ratio=4) == 1, the internal-channel floor every bottleneck already clamps to.
BIT_WIDTH = 8


def _bounded_quant(bit_width: int = BIT_WIDTH, return_quant_tensor: bool = True) -> qnn.QuantHardTanh:
    """Fixed-range [-1, 1] signed quantizer -- see module docstring for why
    this is used instead of QuantENet's usual stats-calibrated
    Int8ActPerTensorFloat for these two boundary points specifically.

    return_quant_tensor must be False wherever this feeds directly into a
    nn.Module's `forward` return value: a QuantTensor is a 4-field named
    tuple (value/scale/zero_point/bit_width), and torch.onnx.export turns
    each field of a *returned* QuantTensor into its own graph output. Those
    extra scale/zero_point/bit_width outputs are compile-time constants
    (fixed HardTanh range), so qonnx's cleanup constant-folds the nodes
    producing them but doesn't drop them from the graph's output list --
    producing a graph whose declared outputs aren't all produced by any
    node (onnx.checker.check_model then fails with "is not an output of
    any node in graph"). Confirmed by isolating a minimal
    QuantHardTanh-only module and export: return_quant_tensor=True there
    reproduces the exact same broken-output symptom, =False does not.
    Everywhere else in this file return_quant_tensor=True is fine (and
    needed, so the next block knows the input's scale) since it's
    consumed by another Brevitas layer, never returned raw from
    `forward`.
    """
    return qnn.QuantHardTanh(
        bit_width=bit_width, min_val=-1.0, max_val=1.0,
        act_quant=Int8ActPerTensorFloat, return_quant_tensor=return_quant_tensor,
    )


class QuantENetDebugEncoder(nn.Module):
    """Encoder-only debug net -- see module docstring."""

    def __init__(self, in_channels: int = 1, channels: int = DEBUG_CHANNELS, bit_width: int = BIT_WIDTH):
        super().__init__()
        self.input_bound = _bounded_quant(bit_width)
        self.initial = QuantInitialBlock(in_channels, channels, bit_width)
        self.down1 = QuantDownsamplingBottleneck(channels, channels, bit_width, dropout_p=0.0)
        self.regular1 = QuantRegularBottleneck(channels, bit_width, dropout_p=0.0)
        self.down2 = QuantDownsamplingBottleneck(channels, channels, bit_width, dropout_p=0.0)
        self.stage2 = QuantRegularBottleneck(channels, bit_width, dropout_p=0.0)
        self.output_bound = _bounded_quant(bit_width, return_quant_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bound(x)
        x = self.initial(x)
        x, _indices1, _size1 = self.down1(x)
        x = self.regular1(x)
        x, _indices2, _size2 = self.down2(x)
        x = self.stage2(x)
        return self.output_bound(x)


class QuantENetDebugFull(nn.Module):
    """Encoder + decoder debug net -- see module docstring. Uses
    upsample_conv (bilinear interpolate), not max_unpool: PyTorch's ONNX
    exporter has no symbolic function for aten::max_unpool2d at all
    (confirmed directly -- torch.onnx.export raises UnsupportedOperatorError
    for it regardless of Brevitas/QONNX), so max_unpool was never actually
    exportable in the first place. QuantENet.py's own __main__ self-test
    export only ever used decoder_type='upsample_conv' for exactly this
    reason, never max_unpool -- this net follows the same only-verified
    path rather than the (uniform-channel-width-enabled but unexportable)
    max_unpool one."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, channels: int = DEBUG_CHANNELS, bit_width: int = BIT_WIDTH):
        super().__init__()
        self.input_bound = _bounded_quant(bit_width)
        self.initial = QuantInitialBlock(in_channels, channels, bit_width)
        self.down1 = QuantDownsamplingBottleneck(channels, channels, bit_width, dropout_p=0.0)
        self.regular1 = QuantRegularBottleneck(channels, bit_width, dropout_p=0.0)
        self.down2 = QuantDownsamplingBottleneck(channels, channels, bit_width, dropout_p=0.0)
        self.stage2 = QuantRegularBottleneck(channels, bit_width, dropout_p=0.0)
        self.up4 = QuantUpsamplingBottleneck(channels, channels, bit_width)
        self.regular4 = QuantRegularBottleneck(channels, bit_width, dropout_p=0.0)
        self.up5 = QuantUpsamplingBottleneck(channels, channels, bit_width)
        self.regular5 = QuantRegularBottleneck(channels, bit_width, dropout_p=0.0)
        self.final = qnn.QuantConvTranspose2d(
            channels, out_channels, kernel_size=2, stride=2, bias=True,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )
        self.output_bound = _bounded_quant(bit_width, return_quant_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bound(x)
        x = self.initial(x)
        x, _indices1, size1 = self.down1(x)
        x = self.regular1(x)
        x, _indices2, size2 = self.down2(x)
        x = self.stage2(x)
        x = self.up4(x, size2, indices=None)
        x = self.regular4(x)
        x = self.up5(x, size1, indices=None)
        x = self.regular5(x)
        x = self.final(x)
        return self.output_bound(x)


def _export_one(model: nn.Module, name: str, dummy: torch.Tensor, run_finn_convert: bool) -> Path:
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    export_qonnx(model, export_path=str(export_path), input_t=dummy)
    qonnx_cleanup(str(export_path), out_file=str(export_path))

    qmodel = ModelWrapper(str(export_path))
    qmodel.set_tensor_datatype(qmodel.graph.input[0].name, DataType["INT8"])
    qmodel.set_tensor_datatype(qmodel.graph.output[0].name, DataType["INT8"])
    qmodel.save(str(export_path))

    import onnx
    onnx.checker.check_model(onnx.load(str(export_path)))
    print(f"{name}: QONNX export OK, passed onnx.checker: {export_path}")

    if run_finn_convert:
        try:
            from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
        except ImportError as error:
            print(
                f"{name}: FINN is not importable in this environment -- skipping ConvertQONNXtoFINN, "
                f"leaving {export_path} as cleaned QONNX only. Import error: {error}"
            )
            return export_path
        finn_model = qmodel.transform(ConvertQONNXtoFINN())
        finn_model.save(str(export_path))
        print(f"{name}: ConvertQONNXtoFINN OK: {export_path}")

    return export_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", type=int, default=DEBUG_CHANNELS, help="Uniform channel width at every stage (default: smallest usable, 4).")
    parser.add_argument("--bit-width", type=int, default=BIT_WIDTH)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2, help="Only used by the encoder+decoder net's final transposed conv.")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"), help="Must be divisible by 8 (initial + down1 + down2, each stride 2).")
    parser.add_argument("--no-finn-convert", action="store_true", help="Skip ConvertQONNXtoFINN even if `finn` is importable (cleaned-QONNX-only output).")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8 (three stride-2 stages).")

    torch.manual_seed(0)
    # Uniform-in-[-1, 1], not true bipolar (+-1 only) -- these are image-like
    # tensors, not the binary feature vectors FINN's cybersec-mlp example
    # bounds to BIPOLAR; continuous-valued [-1, 1] is the image analogue.
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1

    run_finn_convert = not args.no_finn_convert

    encoder = QuantENetDebugEncoder(in_channels=args.in_channels, channels=args.channels, bit_width=args.bit_width).eval()
    with torch.no_grad():
        out = encoder(dummy)
    out_t = out.value if hasattr(out, "value") else out
    print(f"QuantENetDebugEncoder: build+forward OK, output shape {tuple(out_t.shape)}")
    _export_one(encoder, "quantEnet_debug_enc", dummy, run_finn_convert)

    full = QuantENetDebugFull(in_channels=args.in_channels, out_channels=args.out_channels, channels=args.channels, bit_width=args.bit_width).eval()
    with torch.no_grad():
        out = full(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape[2:] == dummy.shape[2:], f"QuantENetDebugFull: output HxW {tuple(out_t.shape[2:])} != input {tuple(dummy.shape[2:])}"
    print(f"QuantENetDebugFull: build+forward OK, output shape {tuple(out_t.shape)}")
    _export_one(full, "quantEnet_debug_full", dummy, run_finn_convert)


if __name__ == "__main__":
    main()
