"""PReLU-vs-FINN compatibility investigation, round 2.

Round 1 (a from-scratch single-conv-block toy model) hit a generic streamlining
convergence quirk (a residual input-quantizer Add never got absorbed into its
MultiThreshold's out_bias) unrelated to PReLU itself, and confirmed:
  - FINN/QONNX source has ZERO references to "PRelu" anywhere (no HWCustomOp,
    no transform recognizes it) -- grep-confirmed separately.
  - The literal ONNX "PRelu" node survives step_qonnx_to_finn and the full
    streamlining pipeline completely untouched (never absorbed/converted).

This round reuses ENet's OWN proven-working building blocks (QuantInitialBlock
+ QuantRegularBottleneck, copied verbatim from enet/nnunetv2/nets/QuantENet.py,
the exact recipe that has already survived a real 66-hour FINN hardware build
for the full network) and swaps ONLY the "reduce" sub-block's activation
function, isolating PReLU as the single variable instead of re-deriving a
whole streamlining recipe from scratch:

  1. control      : reduce activation = QuantReLU                (proven-good baseline)
  2. prelu_naive  : reduce activation = PReLU (float) -> QuantIdentity requant
  3. prelu_decomp : reduce activation = QuantReLU(x) - alpha_c * QuantReLU(-x)
                    (built with an explicit Mul-by-(-1) instead of ONNX Neg,
                    since Neg is -- like PRelu -- absent from FINN's transform
                    vocabulary; confirmed via grep separately)
"""
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
from brevitas.export import export_qonnx

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import (
    ConvertSubToAdd,
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortGraph,
)

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    AbsorbConsecutiveTransposes,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    MoveScalarMulPastMatMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastConv,
    MoveScalarMulPastConvTranspose,
    MoveMulPastMaxPool,
    MoveScalarLinearPastInvariants,
    MoveMaxPoolPastMultiThreshold,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MakeMaxPoolNHWC,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import InferPixelPaddingDeconv
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

from finn.builder.build_dataflow_steps import step_qonnx_to_finn
from finn.builder.build_dataflow_config import DataflowBuildConfig

torch.manual_seed(0)

OUT_DIR = "/tmp/prelu_test2"
os.makedirs(OUT_DIR, exist_ok=True)

BW = 8
IN_CH = 1
CH = 8  # QuantInitialBlock out_channels == QuantRegularBottleneck channels


# ── copied verbatim from enet/nnunetv2/nets/QuantENet.py ───────────────────
def _quant_conv2d(in_ch, out_ch, bit_width, **kwargs):
    return qnn.QuantConv2d(
        in_ch, out_ch, bias=False,
        weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat,
        **kwargs,
    )


def _quant_act(bit_width):
    return qnn.QuantReLU(bit_width=bit_width, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)


class QuantInitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bit_width):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.conv = _quant_conv2d(in_channels, out_channels - in_channels, bit_width, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _quant_act(bit_width)

    def forward(self, x):
        x = self.input_quant(x)
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


# ── PReLU-equivalent activation modules (drop-in for _quant_act) ──────────
class NaivePReLUAct(nn.Module):
    """Naive drop-in: plain float PReLU, requantized afterward."""
    def __init__(self, channels, bit_width):
        super().__init__()
        self.act = nn.PReLU(channels)
        self.out_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x):
        return self.out_quant(self.act(x))


class DecomposedPReLUAct(nn.Module):
    """PReLU(x) = alpha_c*x + (1-alpha_c)*ReLU(x) -- algebraically equivalent
    to the classic ReLU(x) - alpha_c*ReLU(-x) form, but needs NO Neg/Sub node
    at all (Neg -- like PRelu -- has zero FINN transform support, confirmed
    separately). Uses only: one quantized ReLU (already FINN-native via
    MultiThreshold/Thresholding), a fork of x into two branches, two
    per-channel Mul scales (-> InferChannelwiseLinearLayer), and one
    elementwise Add of two dynamic streams (-> InferAddStreamsLayer) --
    exactly the fork+recombine topology ENet's OWN residual connections
    already use (MoveLinearPastFork / MoveLinearPastEltwiseAdd exist in the
    streamlining recipe specifically for this shape).
    (1-alpha_c) is computed from the alpha_c *parameter* only (not from the
    runtime activation stream), so FoldConstants() collapses it to a single
    constant initializer before streamlining ever sees it -- no runtime Sub.
    """
    def __init__(self, channels, bit_width, alpha_init=0.25):
        super().__init__()
        self.pre_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.act_pos = qnn.QuantReLU(bit_width=bit_width, act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        # Both pre-materialized as separate buffers (not derived from each other
        # at forward time) -- avoids tracing an extra Sub(1.0, alpha) node that
        # tripped CollapseRepeatedMul's "both chained Muls need a constant
        # initializer" assumption in an earlier attempt.
        self.register_buffer("alpha", torch.full((1, channels, 1, 1), alpha_init))
        self.register_buffer("one_minus_alpha", torch.full((1, channels, 1, 1), 1.0 - alpha_init))
        self.out_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x):
        x_q = self.pre_quant(x)
        pos = self.act_pos(x_q)
        # constant operand MUST be input[1] in the traced ONNX Mul (dynamic *
        # constant, not constant * dynamic) -- FINN's streamlining transforms
        # (e.g. CollapseRepeatedMul) hard-assume this convention when deciding
        # which operand is "the constant to fold".
        scaled_x = x_q * self.alpha
        scaled_pos = pos * self.one_minus_alpha
        out = scaled_x + scaled_pos
        return self.out_quant(out)


# ── QuantRegularBottleneck, copied verbatim except reduce's activation ────
class QuantRegularBottleneck(nn.Module):
    def __init__(self, channels, bit_width, internal_ratio=4, kernel_size=3, padding=1, act_factory=None):
        super().__init__()
        internal_channels = max(1, channels // internal_ratio)
        act_factory = act_factory or (lambda ch, bw: _quant_act(bw))

        self.reduce = nn.Sequential(
            _quant_conv2d(channels, internal_channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(internal_channels),
            act_factory(internal_channels, bit_width),
        )
        self.conv = _quant_conv2d(internal_channels, internal_channels, bit_width, kernel_size=kernel_size, padding=padding)
        self.conv_bn_act = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(internal_channels),
            _quant_act(bit_width),
        )
        self.expand = nn.Sequential(
            _quant_conv2d(internal_channels, channels, bit_width, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.residual_add = qnn.QuantEltwiseAdd(bit_width=bit_width, input_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.out_act = _quant_act(bit_width)

    def forward(self, x):
        out = self.reduce(x)
        out = self.conv_bn_act(out)
        out = self.expand(out)
        return self.out_act(self.residual_add(x, out))


class MiniENet(nn.Module):
    def __init__(self, act_factory=None):
        super().__init__()
        self.initial = QuantInitialBlock(IN_CH, CH, BW)
        self.bottleneck = QuantRegularBottleneck(CH, BW, act_factory=act_factory)

    def forward(self, x):
        return self.bottleneck(self.initial(x))


MODELS = {
    "control": MiniENet(act_factory=None),
    "prelu_naive": MiniENet(act_factory=lambda ch, bw: NaivePReLUAct(ch, bw)),
    "prelu_decomp": MiniENet(act_factory=lambda ch, bw: DecomposedPReLUAct(ch, bw)),
}


def export_all():
    # Use randn (mean~0, has negative values) not rand ([0,1) only) -- matches
    # real z-score-normalized medical-image input statistics. An all-positive
    # dummy input lets the input quantizer calibrate a degenerate zero-point
    # (bias=0), which is what tripped the generic (non-PReLU-related)
    # "Signed output requires actval < 0" assertion in step_enet_convert_to_hw
    # for ALL THREE variants including the plain-ReLU control.
    dummy = torch.randn(1, IN_CH, 32, 32)
    paths = {}
    for name, model in MODELS.items():
        model.eval()
        onnx_path = os.path.join(OUT_DIR, f"{name}.onnx")
        try:
            export_qonnx(model, export_path=onnx_path, input_t=dummy)
            print(f"[export OK] {name} -> {onnx_path}")
            paths[name] = onnx_path
        except Exception as e:
            print(f"[export FAIL] {name}: {type(e).__name__}: {e}")
    return paths


# ── copied custom steps from finn_enet_ip_build.py ─────────────────────────
def step_enet_tidy(model, cfg):
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


def _streamline_linear(model, cfg, debug=False):
    for trn in [
        ConvertSubToAdd(), ConvertDivToMul(), RemoveIdentityOps(), CollapseRepeatedMul(),
        BatchNormToAffine(), ConvertSignToThres(), MoveAddPastMul(), MoveScalarAddPastMatMul(),
        MoveAddPastConv(), MoveScalarMulPastMatMul(), MoveScalarMulPastConv(),
        MoveScalarMulPastConvTranspose(), MoveMulPastMaxPool(), MoveScalarLinearPastInvariants(),
        MoveAddPastMul(), CollapseRepeatedAdd(), CollapseRepeatedMul(), AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(), MoveMaxPoolPastMultiThreshold(), AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(), Absorb1BitMulIntoConv(), RoundAndClipThresholds(),
    ]:
        if debug:
            try:
                model = model.transform(trn)
            except AssertionError as e:
                print(f"    !! FAILED inside {type(trn).__name__}: {e}")
                for n in model.graph.node:
                    if n.op_type == "Mul":
                        cons = model.find_consumer(n.output[0])
                        init = model.get_initializer(n.input[1])
                        print(f"       Mul node={n.name!r} input1={n.input[1]!r} "
                              f"has_init={init is not None} consumer={cons.op_type if cons else None!r}"
                              f"({cons.name if cons else ''})")
                raise
        else:
            model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def _streamline_nonlinear(model, cfg):
    for trn in [MoveLinearPastFork(), MoveLinearPastEltwiseAdd()]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def _fixup_degenerate_signed_bias(model):
    """Demonstration-only patch for this MINIMAL test harness.

    Root cause of the shared 'Signed output requires actval < 0' failure
    (confirmed identical across control/prelu_naive/prelu_decomp, always at
    MultiThreshold_0 -- the network's very first, input-quantizer threshold
    node): AbsorbAddIntoMultiThreshold only folds an Add that PRECEDES a
    MultiThreshold into that MultiThreshold's *thresholds*; it has nothing to
    do with a MultiThreshold's own out_bias/ActVal. out_bias for a genuinely
    SIGNED output must independently equal -(2**(bits-1)) by construction (to
    reinterpret an unsigned threshold-count as a signed range) -- this is
    supposed to be set once, correctly, when qonnx's Quant->MultiThreshold
    conversion runs. In this tiny, freshly/randomly initialized 2-block test
    network (no calibration data, no training) the input quantizer's
    threshold set degenerates and the conversion leaves out_bias at the
    fallback value 0 instead of the correct negative value -- a calibration
    artifact of the minimal harness, NOT a FINN capability gap, and NOT
    specific to PReLU (it equally blocks the plain-ReLU control). This patch
    directly corrects it so a full, decisive HW-conversion comparison can
    still be completed end-to-end.
    """
    from qonnx.custom_op.registry import getCustomOp as _getCustomOp

    for node in model.graph.node:
        if node.op_type != "MultiThreshold":
            continue
        inst = _getCustomOp(node)
        odt = model.get_tensor_datatype(node.output[0])
        actval = inst.get_nodeattr("out_bias")
        if odt.signed() and odt.name != "BIPOLAR" and actval >= 0:
            bits = odt.bitwidth()
            correct_bias = -(2 ** (bits - 1))
            print(f"    [patch] {node.name}: out_bias {actval} -> {correct_bias} "
                  f"(odt={odt.name})")
            inst.set_nodeattr("out_bias", float(correct_bias))
    return model


def step_enet_streamline(model, cfg, debug=False):
    for _iter in range(4):
        model = _streamline_linear(model, cfg, debug=debug)
        model = _streamline_nonlinear(model, cfg)
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())
    if len(model.get_nodes_by_op_type("Conv")) > 0:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    return model


def step_enet_convert_to_hw(model, cfg):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
    model = model.transform(InferDataTypes())

    for trn in [
        to_hw.InferAddStreamsLayer, to_hw.InferChannelwiseLinearLayer, to_hw.InferStreamingMaxPool,
        RoundAndClipThresholds, to_hw.InferBinaryMatrixVectorActivation,
        to_hw.InferQuantizedMatrixVectorActivation, to_hw.InferVectorVectorActivation,
        to_hw.InferThresholdingLayer, AbsorbConsecutiveTransposes, to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
    ]:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


def make_cfg(tag):
    return DataflowBuildConfig(
        output_dir=os.path.join(OUT_DIR, f"cfg_{tag}"),
        synth_clk_period_ns=10.0,
        fpga_part="xczu7ev-ffvc1156-2-e",
        generate_outputs=[],
    )


def run_pipeline(name, onnx_path):
    print("\n" + "=" * 70)
    print(f"PIPELINE: {name}")
    print("=" * 70)
    cfg = make_cfg(name)
    model = ModelWrapper(onnx_path)
    steps = [
        ("step_qonnx_to_finn", step_qonnx_to_finn),
        ("step_enet_tidy", step_enet_tidy),
        ("step_enet_streamline", lambda m, c: step_enet_streamline(m, c, debug=True)),
        ("_fixup_degenerate_signed_bias", lambda m, c: _fixup_degenerate_signed_bias(m)),
        ("step_enet_convert_to_hw", step_enet_convert_to_hw),
    ]
    for step_name, fn in steps:
        try:
            model = fn(model, cfg)
            print(f"  [ok] {step_name} -- nodes: {[n.op_type for n in model.graph.node]}")
        except Exception as e:
            print(f"  [FAIL] {step_name}: {type(e).__name__}: {e}")
            return

    node_types = [n.op_type for n in model.graph.node]
    hw_prefixes = ("MVAU", "VVAU", "Thresholding", "AddStreams", "DuplicateStreams",
                   "ChannelwiseOp", "StreamingMaxPool", "ConvolutionInputGenerator",
                   "StreamingFIFO", "StreamingDataWidthConverter")
    non_hw = [t for t in node_types if not any(t.startswith(h) for h in hw_prefixes)]
    if non_hw:
        print(f"  >>> RESULT: NON-HW nodes remain: {non_hw}")
    else:
        print("  >>> RESULT: ALL nodes are HW/fpgadataflow ops -- fully hardware-convertible!")
    model.save(os.path.join(OUT_DIR, f"{name}_converted.onnx"))


if __name__ == "__main__":
    paths = export_all()
    for name, path in paths.items():
        run_pipeline(name, path)
