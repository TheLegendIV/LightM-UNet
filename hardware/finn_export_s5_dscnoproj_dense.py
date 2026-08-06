"""Export a FINN-compatible quantized model for the S5-DscNoProjDense
compression-sweep config (stage 5 arch-probe pair):

    config_name = nnUNetTrainerENet_5_1_dscnoprojection_dense_dilation
    channels    = (4, 16, 32, 32, 16, 4)   # 6-tuple: initial,s1,s2,s3,s4,s5
    bnecks      = (4, 8, 8, 2, 1)
    ops_flags   = dilated=1, asymmetric=0, strided=1, dsc=0,
                  context_pattern=dense_dilation, prelu=1, dsc_no_projection=1
                  (all other stage4/5 probe flags = 0)

See compression/config_abbreviations.csv / compression/results.csv for the
source config, and enet/nnunetv2/nets/ENet.py (DSCNoProjectionBottleneck,
DENSE_DILATION_PATTERN, ENet._make_context_stage/_make_shallow_stage) for the
FP32 reference architecture this mirrors.

Unlike hardware/finn_enet_prod_export.py (which sidesteps the PReLU-vs-FINN
problem entirely by using QuantReLU everywhere), this script uses the real
PReLU activation this config trains with, decomposed into FINN-native ops:

    PReLU(x) = alpha_c * x + (1 - alpha_c) * ReLU(x)

(exact algebraic identity, proven both structurally -- reaches full FINN
op-vocabulary parity with a plain-ReLU baseline -- and numerically --
quantization error statistically identical to naive-requantized real PReLU;
see hardware/README.md's "PReLU vs. QuantReLU -- FINN compatibility
investigation" section and hardware/_tmp_prelu_investigation2.py) wherever
the FP32 architecture would use PReLU: the encoder half only (initial,
down1/down2, regular1/stage2/stage3's DSC-no-projection blocks) -- the
decoder half (up4/up5, regular4/regular5) always uses plain ReLU in this
architecture regardless of the prelu flag, matching ENet.py's own
hardcoded relu=True on that half.

"Empty" model: randomly initialized, untrained (same convention as
hardware/finn_enet_prod_export.py and hardware/_tmp_prelu_investigation2.py)
-- this only tests structural FINN convertibility, not accuracy.

Usage (run inside the pytorch training container, which already has
torch/brevitas/qonnx -- NOT inside the FINN container, which lacks a
straight `pip install` of this repo's own dependencies):
    python hardware/finn_export_s5_dscnoproj_dense.py

Output: hardware/outputs/finn_exports/quantEnet_s5_dscnoproj_dense_int8.onnx
Then, inside the FINN container:
    docker cp hardware/outputs/finn_exports/quantEnet_s5_dscnoproj_dense_int8.onnx \
        <finn_container_id>:/home/thelegendiv/finn/notebooks/enet/
    docker exec <finn_container_id> python /tmp/finn_enet_build.py quantEnet_s5_dscnoproj_dense_int8
(finn_enet_build.py's estimate-only step list stops at
step_generate_estimate_reports -- i.e. resource/cycle ESTIMATES only, no
real HLS/RTL codegen or synthesis, matching "until the hardware generation
part".)
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
from nnunetv2.nets.QuantENet import _quant_conv2d, _quant_act  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "finn_exports"
DEFAULT_CHANNELS = (4, 16, 32, 32, 16, 4)   # initial, s1, s2, s3, s4, s5
DEFAULT_BNECKS = (4, 8, 8, 2, 1)
BIT_WIDTH = 8
# ENet.py's DENSE_DILATION_PATTERN flattened to its dilation values --
# every context-stage slot dilated (2/4/8/16 schedule repeated twice),
# no plain/asymmetric slots at all.
DENSE_DILATIONS = [2, 4, 8, 16, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Decomposed PReLU (proven in hardware/_tmp_prelu_investigation2.py)
# ---------------------------------------------------------------------------

class DecomposedPReLUAct(nn.Module):
    """PReLU(x) = alpha_c*x + (1-alpha_c)*ReLU(x), exported as a plain ONNX
    `LeakyRelu(alpha)` sandwiched between two int8 Quant identities:

        pre_quant (Quant) -> LeakyRelu(alpha) -> out_quant (Quant)

    v3: no fork, no second Conv branch, no dynamic Add-join. Two earlier
    approaches were tried and BOTH failed identically at FINN's
    step_create_dataflow_partition ("cycle-free graph violated: partition
    depends on itself"):
      v1 (raw float buffers, x_q*alpha + pos*(1-alpha)) and
      v2 (two depthwise 1x1 QuantConv2d branches instead of raw Mul)
    both produce the same fork -> 2 branches -> dynamic Add-join -> Quant
    topology. The un-absorbable float scale (whether a raw Python constant
    or a Conv's own weight-dequant scale) always ends up feeding an Add with
    TWO dynamic inputs, and no FINN transform absorbs a Mul/Add into a
    MultiThreshold across a join like that (AbsorbMulIntoMultiThreshold /
    AbsorbAddIntoMultiThreshold both require a direct, non-fork/join Mul/Add
    -> MultiThreshold edge).

    This version sidesteps the problem instead of trying to patch around it:
    PReLU(x) = x if x>=0 else alpha*x is a monotonic int8->int8 step
    function, and MultiThreshold can represent ANY monotonic step function
    exactly via non-uniformly-spaced thresholds -- no fork/Add needed at
    all. The exported ONNX graph here is just Quant -> LeakyRelu -> Quant;
    the FINN build (hardware/finn_enet_build_decomposed_prelu.py's
    step_fuse_leaky_relu_to_threshold) then fuses
    [MultiThreshold(pre_quant) -> LeakyRelu -> MultiThreshold(out_quant)]
    into a SINGLE MultiThreshold node after step_qonnx_to_finn converts the
    Quant nodes: since leaky_relu is monotonic and invertible for
    0<alpha<1, a threshold t meant to compare against leaky_relu(v) can be
    pulled back to compare directly against v via
    `t if t>=0 else t/alpha` -- an exact fusion, not an approximation.
    `pre_quant` uses return_quant_tensor=False since its output only feeds a
    plain (non-Brevitas-aware) torch.nn.functional.leaky_relu call."""

    def __init__(self, channels: int, bit_width: int, alpha_init: float = 0.25):
        super().__init__()
        self.pre_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.alpha_init = alpha_init
        self.out_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.pre_quant(x)
        out = torch.nn.functional.leaky_relu(x_q, negative_slope=self.alpha_init)
        return self.out_quant(out)


def _decomposed_prelu_factory(channels: int, bit_width: int) -> nn.Module:
    return DecomposedPReLUAct(channels, bit_width)


def _plain_relu_factory(channels: int, bit_width: int) -> nn.Module:
    return _quant_act(bit_width)


# ---------------------------------------------------------------------------
# FINN-compatible block definitions (encoder blocks take an act_factory so
# they can use either the decomposed PReLU (encoder, per ops_flags prelu=1)
# or plain QuantReLU (decoder, always -- matches ENet.py's hardcoded
# relu=True on regular4/regular5/up4/up5 regardless of the prelu flag))
# ---------------------------------------------------------------------------

class FINNInitialBlock(nn.Module):
    """Single-conv initial block (no MaxPool branch -> no Concat -> FINN-safe),
    same simplification as hardware/finn_enet_prod_export.py's FINNInitialBlock."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, act_factory):
        super().__init__()
        self.conv = _quant_conv2d(in_ch, out_ch, bit_width, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_factory(out_ch, bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FINNDownsamplingBottleneck(nn.Module):
    """Downsampling bottleneck without MaxPool-with-indices (matches
    hardware/finn_enet_prod_export.py's FINNDownsamplingBottleneck) --
    shortcut = MaxPool(no indices) + 1x1 projection, branch = stride-2 2x2
    conv + 3x3 conv + 1x1 expand, plain `+` residual (both operands
    pre-quantized to INT8/UINT8 via act_factory, not brevitas
    QuantEltwiseAdd)."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, act_factory,
                 dropout_p: float = 0.01, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, out_ch // 4)

        self.shortcut_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.shortcut_proj = nn.Sequential(
            _quant_conv2d(in_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            act_factory(out_ch, bit_width),
        )
        self.reduce = nn.Sequential(
            _quant_conv2d(in_ch, internal_ch, bit_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.conv = nn.Sequential(
            _quant_conv2d(internal_ch, internal_ch, bit_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(internal_ch),
            act_factory(internal_ch, bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_ch, out_ch, bit_width, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            act_factory(out_ch, bit_width),  # quantize before residual Add
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(out_ch, bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_proj(self.shortcut_pool(x)) if self.residual else None
        out = self.dropout(self.expand(self.conv(self.reduce(x))))
        if self.residual:
            out = out + shortcut
        return self.act(out)


class FINNUpsamplingBottleneck(nn.Module):
    """Main path = QuantConvTranspose2d directly (no F.interpolate/Resize, no
    MaxUnpool) -- matches hardware/finn_enet_prod_export.py's
    FINNUpsamplingBottleneck exactly. Decoder-half block: always plain
    QuantReLU (never PReLU), matching ENet.py's UpsamplingBottleneck
    hardcoding relu=True regardless of the network's prelu flag."""

    def __init__(self, in_ch: int, out_ch: int, bit_width: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        internal_ch = max(1, in_ch // 4)

        self.main_up = qnn.QuantConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )
        self.main_bn = nn.BatchNorm2d(out_ch)
        self.main_act = _quant_act(bit_width)

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
            _quant_act(bit_width),
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.act = _quant_act(bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_act(self.main_bn(self.main_up(x)))
        out = self.dropout(self.expand(self.up(self.reduce(x))))
        if self.residual:
            out = out + main
        return self.act(out)


class FINNDSCNoProjectionBottleneck(nn.Module):
    """FINN-safe mirror of ENet.py's DSCNoProjectionBottleneck: depthwise KxK
    (groups=channels, dilation-aware) + pointwise 1x1, at full `channels`
    width throughout -- NO reduce/expand projection pair at all (unlike
    RegularBottleneck/FINNRegularBottleneck). Used (per dsc_no_projection=1)
    for regular1/stage2/stage3/regular4/regular5 -- the pattern's dilation
    schedule (dense_dilation for stage2/3, none for regular1/4/5) is the
    caller's responsibility via the `dilation` arg.

    An extra act_factory quantizer is inserted after the pointwise conv's BN
    (the FP32 DSCNoProjectionBottleneck's `conv` Sequential has no
    activation there) so both operands of the residual Add are already
    quantized -- same "quantize to INT8 before a plain `+`" convention
    hardware/finn_enet_prod_export.py uses everywhere else, instead of
    Brevitas's QuantEltwiseAdd module."""

    def __init__(self, channels: int, bit_width: int, act_factory,
                 dilation: int = 1, dropout_p: float = 0.1, residual: bool = True):
        super().__init__()
        self.residual = residual
        padding = dilation
        self.conv = nn.Sequential(
            _quant_conv2d(channels, channels, bit_width, kernel_size=3, padding=padding,
                          dilation=dilation, groups=channels),
            nn.BatchNorm2d(channels),
            act_factory(channels, bit_width),
            _quant_conv2d(channels, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
            act_factory(channels, bit_width),  # quantize before residual Add
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = act_factory(channels, bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv(x))
        if self.residual:
            out = out + x
        return self.act(out)


class FINNQuantENetS5DscNoProjDense(nn.Module):
    """FINN-compatible quantized model for the S5-DscNoProjDense compression-
    sweep config (nnUNetTrainerENet_5_1_dscnoprojection_dense_dilation):
    dsc_no_projection=1 (DSC-everywhere, no reduce/expand, on regular1/
    stage2/stage3/regular4/regular5) + context_pattern=dense_dilation
    (every context-stage slot dilated, 2/4/8/16 x2) + prelu=1 (decomposed,
    encoder-half only) + strided=1 + asymmetric=0.

    channels is a 6-tuple (initial, s1, s2, s3, s4, s5) -- this config has
    s2==s3==32 so proj2_to_3 collapses to nn.Identity, but the 6-tuple form
    is kept general (matches ENet.py's own 6-tuple support)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 5,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        bottlenecks_per_stage: tuple[int, ...] = DEFAULT_BNECKS,
        bit_width: int = BIT_WIDTH,
        residual: bool = True,
    ):
        super().__init__()
        if len(channels) != 6:
            raise ValueError(
                "channels must be a 6-tuple (initial, stage1, stage2, stage3, stage4, "
                f"stage5) for this DSC-no-projection/dense-dilation config, got {channels!r}."
            )
        if len(bottlenecks_per_stage) != 5:
            raise ValueError("bottlenecks_per_stage must have 5 values (stage1, stage2, stage3, regular4, regular5).")
        c0, c1, c2, c3, c4, c5 = channels
        n1, n2, n3, n4, n5 = bottlenecks_per_stage

        self.initial = FINNInitialBlock(in_channels, c0, bit_width, _decomposed_prelu_factory)

        self.down1 = FINNDownsamplingBottleneck(c0, c1, bit_width, _decomposed_prelu_factory,
                                                 dropout_p=0.01, residual=residual)
        self.regular1 = nn.Sequential(*[
            FINNDSCNoProjectionBottleneck(c1, bit_width, _decomposed_prelu_factory,
                                          dilation=1, dropout_p=0.01, residual=residual)
            for _ in range(n1)
        ])

        self.down2 = FINNDownsamplingBottleneck(c1, c2, bit_width, _decomposed_prelu_factory,
                                                 dropout_p=0.1, residual=residual)
        self.stage2 = self._make_dense_context_stage(c2, n2, bit_width, residual)
        self.proj2_to_3 = (
            nn.Identity()
            if c2 == c3
            else nn.Sequential(
                _quant_conv2d(c2, c3, bit_width, kernel_size=1),
                nn.BatchNorm2d(c3),
                _decomposed_prelu_factory(c3, bit_width),
            )
        )
        self.stage3 = self._make_dense_context_stage(c3, n3, bit_width, residual)

        # Decoder half: plain ReLU always (ENet.py hardcodes relu=True here
        # regardless of the network's prelu flag) -- no PReLU decomposition
        # needed, FINNUpsamplingBottleneck/_plain_relu_factory already emit
        # native QuantReLU.
        self.up4 = FINNUpsamplingBottleneck(c3, c4, bit_width, residual=residual)
        self.regular4 = nn.Sequential(*[
            FINNDSCNoProjectionBottleneck(c4, bit_width, _plain_relu_factory,
                                          dilation=1, dropout_p=0.1, residual=residual)
            for _ in range(n4)
        ])
        self.up5 = FINNUpsamplingBottleneck(c4, c5, bit_width, residual=residual)
        self.regular5 = nn.Sequential(*[
            FINNDSCNoProjectionBottleneck(c5, bit_width, _plain_relu_factory,
                                          dilation=1, dropout_p=0.1, residual=residual)
            for _ in range(n5)
        ])

        # Final: no bias (bias in ConvTranspose requires extra BN/threshold
        # handling), same as finn_enet_prod_export.py.
        self.final = qnn.QuantConvTranspose2d(
            c5, out_channels, kernel_size=2, stride=2, bias=False,
            weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        )

    @staticmethod
    def _make_dense_context_stage(channels: int, n: int, bit_width: int, residual: bool) -> nn.Sequential:
        blocks = []
        for i in range(n):
            dilation = DENSE_DILATIONS[i % len(DENSE_DILATIONS)]
            blocks.append(FINNDSCNoProjectionBottleneck(
                channels, bit_width, _decomposed_prelu_factory,
                dilation=dilation, dropout_p=0.1, residual=residual,
            ))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.regular1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.proj2_to_3(x)
        x = self.stage3(x)
        x = self.regular4(self.up4(x))
        x = self.regular5(self.up5(x))
        out = self.final(x)
        return out.value if hasattr(out, "value") else out


# ---------------------------------------------------------------------------
# Export helpers (identical convention to finn_enet_prod_export.py)
# ---------------------------------------------------------------------------

def _fast_cleanup(model):
    """Equivalent to qonnx.util.cleanup.cleanup_model(), but replaces the
    default FoldConstants with FoldConstantsFiltered.

    qonnx's stock FoldConstants (see qonnx/transformation/fold_constants.py)
    deliberately folds only ONE constant-only node per apply() call, then
    triggers a full-graph InferShapes() pass, relying on ModelWrapper's
    transform() wrapper looping apply() until no more folds are found
    (a workaround for https://github.com/fastmachinelearning/qonnx/issues/104).
    For a plain QuantReLU network this is a handful of folds and is fast. Our
    encoder's decomposed PReLU (DecomposedPReLUAct, ~2-3x the activation
    count of a QuantReLU-only network of the same depth) multiplies the
    number of constant-only subgraphs (each Quant node's scale/zero-point
    computation) several-fold -- with ~1100 total nodes this turned into
    100+ sequential O(n) InferShapes() calls (~9.5s each), i.e. 15-30+
    minutes just for this one step (confirmed via instrumented timing on this
    exact model). FoldConstantsFiltered folds ALL currently-eligible nodes in
    a single pass before its own single InferShapes() call, cutting this to a
    handful of passes regardless of total foldable-node count.
    """
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.general import (
        GiveUniqueParameterTensors, RemoveUnusedTensors, RemoveStaticGraphInputs,
        GiveUniqueNodeNames, GiveReadableTensorNames,
    )
    from qonnx.transformation.fold_constants import FoldConstantsFiltered
    from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit

    preserve_qnt_optypes = ["Quant", "BipolarQuant", "QuantizeLinear", "DequantizeLinear"]

    def _foldable(model, node):
        return node.op_type not in preserve_qnt_optypes

    # temporary fix for QONNX op domains (mirrors qonnx.util.cleanup.cleanup_model)
    for q_op_type in ["Quant", "Trunc", "BipolarQuant"]:
        for qnt_node in model.get_nodes_by_op_type(q_op_type):
            qnt_node.domain = "qonnx.custom_op.general"

    for t in [
        InferShapes(),
        GiveUniqueParameterTensors(),
        FoldConstantsFiltered(_foldable),
        FoldTransposeIntoQuantInit(),
        RemoveUnusedTensors(),
        RemoveStaticGraphInputs(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]:
        model = model.transform(t)
    return model


def export_model(model: nn.Module, name: str, dummy: torch.Tensor) -> Path:
    """Export model to cleaned QONNX, set INT8 datatypes, verify it loads."""
    from brevitas.export import export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.datatype import DataType
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.onnx"

    model.cpu().eval()
    export_qonnx(model, export_path=str(out_path), input_t=dummy)

    qm = ModelWrapper(str(out_path))
    qm = _fast_cleanup(qm)
    qm.save(str(out_path))

    qm = ModelWrapper(str(out_path))
    qm.set_tensor_datatype(qm.graph.input[0].name, DataType["INT8"])
    qm.set_tensor_datatype(qm.graph.output[0].name, DataType["INT8"])
    qm.save(str(out_path))

    loaded = onnx.load(str(out_path))
    assert len(loaded.graph.node) > 0, "exported model has no nodes"

    ops: dict[str, int] = {}
    for n in qm.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {name}: {len(qm.graph.node)} nodes -- {dict(sorted(ops.items()))}")
    print(f"  Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", type=int, nargs=6, default=list(DEFAULT_CHANNELS), metavar="C",
                        help="Channel widths per stage, 6 values (default: 4 16 32 32 16 4)")
    parser.add_argument("--bnecks", type=int, nargs=5, default=list(DEFAULT_BNECKS), metavar="N",
                        help="Bottleneck counts per stage (default: 4 8 8 2 1)")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=5,
                        help="Segmentation classes (default 5 = background + 4 ARCADE classes)")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(64, 64), metavar=("H", "W"),
                        help="Spatial dims for export dummy input (default: 64 64). Must be divisible by 8.")
    parser.add_argument("--bit-width", type=int, default=BIT_WIDTH)
    parser.add_argument("--no-residuals", action="store_true", help="Remove all residual shortcuts.")
    args = parser.parse_args()

    h, w = args.input_hw
    if h % 8 != 0 or w % 8 != 0:
        parser.error(f"--input-hw {h}x{w}: both dims must be divisible by 8.")

    torch.manual_seed(0)
    dummy = torch.rand(1, args.in_channels, h, w) * 2 - 1

    residual = not args.no_residuals
    channels = tuple(args.channels)
    bnecks = tuple(args.bnecks)
    bw = args.bit_width

    print("\nFINNQuantENetS5DscNoProjDense export")
    print(f"  channels={channels}, bnecks={bnecks}")
    print(f"  input=({args.in_channels},{h},{w})  bit_width={bw}")
    print(f"  residual={residual}  (dsc_no_projection=1, context_pattern=dense_dilation, prelu=1(decomposed), asymmetric=0, strided=1)")

    model = FINNQuantENetS5DscNoProjDense(
        in_channels=args.in_channels, out_channels=args.out_channels,
        channels=channels, bottlenecks_per_stage=bnecks,
        bit_width=bw, residual=residual,
    ).eval()

    with torch.no_grad():
        out = model(dummy)
    out_t = out.value if hasattr(out, "value") else out
    assert out_t.shape[2:] == (h, w), f"output HxW {tuple(out_t.shape[2:])} != input ({h},{w})"
    assert out_t.shape[1] == args.out_channels, f"output channels {out_t.shape[1]} != {args.out_channels}"
    print(f"  forward OK: output shape {tuple(out_t.shape)}")

    suffix = "_no_res" if not residual else ""
    export_model(model, f"quantEnet_s5_dscnoproj_dense_int{bw}{suffix}", dummy)

    print("\nDone. Copy to FINN container with:")
    print(f"  docker cp hardware/outputs/finn_exports/quantEnet_s5_dscnoproj_dense_int{bw}{suffix}.onnx "
          f"<container_id>:/home/thelegendiv/finn/notebooks/enet/")
    print(f"  docker exec <container_id> python /tmp/finn_enet_build.py quantEnet_s5_dscnoproj_dense_int{bw}{suffix}")


if __name__ == "__main__":
    main()
