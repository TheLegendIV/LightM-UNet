"""FINN estimation build for QuantENet configs that use the decomposed-PReLU
activation (PReLU(x) = alpha*x + (1-alpha)*ReLU(x)) -- currently just
S5-DscNoProjDense (quantEnet_s5_dscnoproj_dense_int8).

Identical to finn_enet_build.py's pipeline (imported directly, not
duplicated) except for one extra step inserted between step_enet_streamline
and step_enet_convert_to_hw: _fixup_degenerate_signed_bias.

Why this extra step is needed here (and not for finn_enet_build.py's other
QuantReLU-only configs, e.g. E1/O2/O8): this model's DecomposedPReLUAct uses
a *signed* Int8 quantizer (QuantIdentity w/ Int8ActPerTensorFloat) on its
pre/out activations -- the ReLU-based QuantReLU activations used everywhere
else in the codebase are unsigned (Uint8ActPerTensorFloat) and never hit
this bug. On this randomly-initialized/untrained ("empty") network, the
very first such signed quantizer (MultiThreshold_0, the input-side act of
the first DecomposedPReLUAct in FINNInitialBlock) gets a degenerate
calibration where qonnx's Quant->MultiThreshold conversion leaves
out_bias=0 instead of the correct -(2**(bits-1)), which trips FINN's
`assert (not odt.signed()) or (actval < 0)` in convert_to_hw_layers.py.
Exact same root cause + fix already proven in
hardware/_tmp_prelu_investigation2.py's minimal 2-block test harness; see
that file's _fixup_degenerate_signed_bias docstring for the full analysis.

Run inside the FINN container:
    docker exec <container_id> python /home/thelegendiv/finn/notebooks/enet/finn_enet_build_decomposed_prelu.py quantEnet_s5_dscnoproj_dense_int8
"""

import os
import sys

import numpy as np
import onnx.helper as oh
from onnx.helper import get_attribute_value

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

# Reuse the exact, proven tidy/streamline/convert_to_hw step implementations
# (and all their sys.path/PATH setup) from the production build script --
# importing does NOT execute its `if __name__ == "__main__"` build call.
# MUST come before any qonnx/finn imports below: this is what inserts
# /home/thelegendiv/finn/src, deps/qonnx/src, deps/brevitas/src etc. onto
# sys.path (finn is a source install, not pip-installed).
from finn_enet_build import (
    ENET_DIR,
    step_enet_tidy,
    step_enet_streamline,
    step_enet_convert_to_hw,
)

from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.transformation.streamline.reorder import (
    MoveScalarMulPastMatMul,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedAdd,
)
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbTransposeIntoMultiThreshold,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_s5_dscnoproj_dense_int8"
MODEL_FILE = os.path.join(ENET_DIR, f"{MODEL_NAME}.onnx")

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ENET_DIR, "finn_deployment_outputs", f"estimates_{MODEL_NAME}_{timestamp}")


def _fixup_degenerate_signed_bias(model: ModelWrapper, cfg: DataflowBuildConfig = None) -> ModelWrapper:
    """Copied verbatim (module-load-safe signature) from
    hardware/_tmp_prelu_investigation2.py -- see that file for the full
    root-cause analysis. Corrects any signed-output MultiThreshold node
    whose out_bias attribute was left at the degenerate fallback 0 instead
    of the correct -(2**(bits-1)); a no-op for any node that's already
    correct, so safe to apply unconditionally."""
    for node in model.graph.node:
        if node.op_type != "MultiThreshold":
            continue
        inst = getCustomOp(node)
        odt = model.get_tensor_datatype(node.output[0])
        actval = inst.get_nodeattr("out_bias")
        if odt.signed() and odt.name != "BIPOLAR" and actval >= 0:
            bits = odt.bitwidth()
            correct_bias = -(2 ** (bits - 1))
            print(f"    [fixup] {node.name}: out_bias {actval} -> {correct_bias} (odt={odt.name})")
            inst.set_nodeattr("out_bias", float(correct_bias))
    return model


def step_fuse_leaky_relu_to_threshold(model: ModelWrapper, cfg: DataflowBuildConfig = None) -> ModelWrapper:
    """Replace [MultiThreshold(pre_quant) -> LeakyRelu(alpha) -> MultiThreshold(out_quant)]
    with a SINGLE MultiThreshold node. Must run right after step_qonnx_to_finn
    converts the surrounding Quant nodes into MultiThreshold, before
    streamlining gets a chance to touch the LeakyRelu it doesn't understand.

    Why this is needed: DecomposedPReLUAct (v3) is exported as
    `pre_quant(Quant) -> torch.nn.functional.leaky_relu -> out_quant(Quant)`
    specifically to AVOID the fork/2-branch/dynamic-Add-join topology that
    two earlier implementations (v1: raw float buffers, v2: depthwise
    QuantConv2d branches) both produced -- both failed identically at
    step_create_dataflow_partition ("cycle-free graph violated: partition
    depends on itself") because no FINN transform absorbs a Mul/Add into a
    MultiThreshold across a dynamic-dynamic Add join. LeakyRelu itself has
    no FINN HW-layer conversion rule (it isn't Thresholding/Conv/elementwise
    -affine), so it must be eliminated before step_enet_convert_to_hw runs.

    The fusion is EXACT, not an approximation. qonnx's MultiThreshold
    computes `count = #{thresholds t : v >= t}`, `out = count*out_scale +
    out_bias` (qonnx/custom_op/general/multithreshold.py). leaky_relu(v) = v
    if v>=0 else alpha*v is strictly increasing for 0<alpha<1, hence
    invertible, so for the fused node to satisfy
    `count_fused(v_in) == count_out(leaky_relu(v_in))` for every v_in, each
    of consumer's threshold values t just needs to be replaced by
    `leaky_relu^-1(t) = t if t>=0 else t/alpha` -- monotonicity means the
    ">=" comparison commutes exactly through the inverse. out_scale/
    out_bias/out_dtype/data_layout are copied unchanged from the original
    consumer MultiThreshold; the fused node's input is simply producer's
    existing output tensor -- no rewiring of anything upstream needed.
    """
    graph = model.graph
    fused = 0
    for node in list(graph.node):
        if node.op_type != "LeakyRelu":
            continue
        alpha = None
        for a in node.attribute:
            if a.name == "alpha":
                alpha = a.f
        assert alpha is not None and 0.0 < alpha < 1.0, f"unexpected LeakyRelu alpha={alpha}"

        if model.is_fork_node(node) or model.is_join_node(node):
            continue  # not the simple 1-in-1-out chain we expect; leave alone

def _walk_back_affine_to_multithreshold(model, tensor_name, max_hops=6):
    """Starting at `tensor_name`, walk backward through a chain of scalar
    Add/Mul nodes (each single-producer/single-consumer, i.e. not a
    fork/join) until reaching a MultiThreshold node. Returns
    `(multithreshold_node, scale, bias, chain_nodes)` such that
    `tensor_value == multithreshold_raw_output * scale + bias`
    (`chain_nodes` lists the Add/Mul nodes walked through, closest-to-
    MultiThreshold first), or None if no such chain is found within
    max_hops (unexpected op type, fork/join, non-scalar constant, etc).

    This exists because qonnx's Quant->MultiThreshold conversion leaves the
    dequantization affine transform (real_value = raw_count*scale + bias)
    as separate trailing Add/Mul nodes rather than folded into
    MultiThreshold's own out_scale/out_bias attributes -- that folding is
    normally done by streamlining's AbsorbSignBiasIntoMultiThreshold, which
    hasn't run yet at the point this fusion step executes (it must run
    BEFORE step_enet_streamline, since streamlining doesn't know how to
    move anything past a LeakyRelu)."""
    chain = []  # (op_type, scalar_value), closest-to-tensor_name first
    cur = tensor_name
    for _ in range(max_hops):
        producer = model.find_producer(cur)
        if producer is None:
            return None
        if producer.op_type == "MultiThreshold":
            scale, bias = 1.0, 0.0
            for op_type, a_val, _n in reversed(chain):
                if op_type == "Add":
                    bias = bias + a_val
                else:  # Mul
                    scale = scale * a_val
                    bias = bias * a_val
            return producer, scale, bias, [c[2] for c in chain]
        if producer.op_type not in ("Add", "Mul"):
            return None
        if model.is_fork_node(producer) or model.is_join_node(producer):
            return None
        const = model.get_initializer(producer.input[1])
        data_input = producer.input[0]
        if const is None:
            const = model.get_initializer(producer.input[0])
            data_input = producer.input[1]
        if const is None or np.asarray(const).size != 1:
            return None
        chain.append((producer.op_type, float(np.asarray(const).reshape(-1)[0]), producer))
        cur = data_input
    return None


def step_fuse_leaky_relu_to_threshold(model: ModelWrapper, cfg: DataflowBuildConfig = None) -> ModelWrapper:
    """Replace [MultiThreshold(pre_quant) -> (dequant Add/Mul chain) ->
    LeakyRelu(alpha) -> MultiThreshold(out_quant)] with a SINGLE
    MultiThreshold node. Must run right after step_qonnx_to_finn converts
    the surrounding Quant nodes into MultiThreshold, before streamlining
    gets a chance to touch the LeakyRelu it doesn't understand.

    Why this is needed: DecomposedPReLUAct (v3) is exported as
    `pre_quant(Quant) -> torch.nn.functional.leaky_relu -> out_quant(Quant)`
    specifically to AVOID the fork/2-branch/dynamic-Add-join topology that
    two earlier implementations (v1: raw float buffers, v2: depthwise
    QuantConv2d branches) both produced -- both failed identically at
    step_create_dataflow_partition ("cycle-free graph violated: partition
    depends on itself") because no FINN transform absorbs a Mul/Add into a
    MultiThreshold across a dynamic-dynamic Add join. LeakyRelu itself has
    no FINN HW-layer conversion rule (it isn't Thresholding/Conv/elementwise
    -affine), so it must be eliminated before step_enet_convert_to_hw runs.

    In practice, right after step_qonnx_to_finn, pre_quant's MultiThreshold
    is followed by a small dequantization chain (observed: Add(zero-point)
    then Mul(scale)) before it reaches the LeakyRelu -- qonnx's Quant
    conversion leaves this un-folded until streamlining's
    AbsorbSignBiasIntoMultiThreshold runs, which is too late for us here.
    `_walk_back_affine_to_multithreshold` walks through that chain and
    returns the equivalent `raw_count -> scale, bias` affine so it can be
    folded directly into the new fused thresholds.

    The fusion is EXACT, not an approximation. qonnx's MultiThreshold
    computes `count = #{thresholds t : v >= t}`, `out = count*out_scale +
    out_bias` (qonnx/custom_op/general/multithreshold.py). leaky_relu(v) = v
    if v>=0 else alpha*v is strictly increasing for 0<alpha<1, hence
    invertible: leaky_relu^-1(t) = t if t>=0 else t/alpha. Given the dequant
    affine v = raw_count*scale + bias (scale > 0, a real quantization step
    size), for the fused node to satisfy
    `count_fused(raw_count) == count_out(leaky_relu(v))` for every
    raw_count, each of consumer's threshold values t must become
    `(leaky_relu^-1(t) - bias) / scale` -- monotonicity of both leaky_relu
    and the (scale>0) affine means every ">=" comparison commutes exactly
    through both inverses. out_scale/out_bias/out_dtype/data_layout are
    copied unchanged from the original consumer MultiThreshold; the fused
    node's input is simply the producer MultiThreshold's raw output tensor
    -- no rewiring of anything upstream needed.
    """
    graph = model.graph
    fused = 0
    for node in list(graph.node):
        if node.op_type != "LeakyRelu":
            continue
        alpha = None
        for a in node.attribute:
            if a.name == "alpha":
                alpha = a.f
        assert alpha is not None and 0.0 < alpha < 1.0, f"unexpected LeakyRelu alpha={alpha}"

        if model.is_fork_node(node) or model.is_join_node(node):
            continue  # not the simple 1-in-1-out chain we expect; leave alone

        consumer = model.find_consumer(node.output[0])
        if consumer is None or consumer.op_type != "MultiThreshold":
            continue

        walk = _walk_back_affine_to_multithreshold(model, node.input[0])
        if walk is None:
            continue
        producer, scale, bias, chain_nodes = walk
        if scale <= 0:
            continue  # our formula assumes a positive dequant scale

        cons_inst = getCustomOp(consumer)
        thresholds = model.get_initializer(consumer.input[1])
        assert thresholds is not None
        leaky_inv = np.where(thresholds >= 0, thresholds, thresholds / alpha)
        new_thresholds = ((leaky_inv - bias) / scale).astype(thresholds.dtype)

        new_thresh_name = model.make_new_valueinfo_name()
        model.set_initializer(new_thresh_name, new_thresholds)

        fused_node = oh.make_node(
            "MultiThreshold",
            [producer.output[0], new_thresh_name],
            [consumer.output[0]],
            domain=consumer.domain,
            out_dtype=cons_inst.get_nodeattr("out_dtype"),
            out_scale=cons_inst.get_nodeattr("out_scale"),
            out_bias=cons_inst.get_nodeattr("out_bias"),
            data_layout=cons_inst.get_nodeattr("data_layout"),
        )

        for n2 in chain_nodes + [node, consumer]:
            graph.node.remove(n2)
        graph.node.append(fused_node)
        fused += 1

    if fused:
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    print(f"    [fuse_leaky_relu] fused {fused} LeakyRelu(s) into single MultiThreshold ops")
    return model


def _move_scalar_op_past_im2col(model: ModelWrapper, op_type: str):
    """Move a scalar Mul/Add past a directly-following Im2Col node. Im2Col
    is a pure data-rearrangement op (patch extraction -- every output
    element is literally one input element, no combination across
    positions/channels), so a scalar affine commutes through it exactly:
    Im2Col(scale*x + bias) == scale*Im2Col(x) + bias, with the SAME scalar
    constant on both sides (unlike MoveScalarMulPastMatMul/
    MoveScalarAddPastMatMul, which need a dot-product correction because
    MatMul actually combines values)."""
    graph = model.graph
    graph_modified = False
    for n in list(graph.node):
        if n.op_type != op_type:
            continue
        if model.is_fork_node(n) or model.is_join_node(n):
            continue
        consumer = model.find_consumer(n.output[0])
        if consumer is None or consumer.op_type != "Im2Col" or model.is_join_node(consumer):
            continue
        A = model.get_initializer(n.input[1])
        if A is None or np.prod(A.shape) != 1:
            continue
        start_name = n.input[0]
        scalar_name = n.input[1]
        middle_name = n.output[0]
        end_name = consumer.output[0]
        attrs = {a.name: get_attribute_value(a) for a in consumer.attribute}
        new_im2col = oh.make_node(
            "Im2Col", [start_name], [middle_name], name=consumer.name, domain=consumer.domain, **attrs
        )
        new_scalar_op = oh.make_node(op_type, [middle_name, scalar_name], [end_name], name=n.name)
        node_ind = list(graph.node).index(n)
        graph.node.remove(n)
        graph.node.remove(consumer)
        graph.node.insert(node_ind, new_im2col)
        graph.node.insert(node_ind + 1, new_scalar_op)
        graph_modified = True
    if graph_modified:
        model = model.transform(InferShapes())
    return model, graph_modified


class MoveScalarMulPastIm2Col(Transformation):
    """See _move_scalar_op_past_im2col."""

    def apply(self, model):
        return _move_scalar_op_past_im2col(model, "Mul")


class MoveScalarAddPastIm2Col(Transformation):
    """See _move_scalar_op_past_im2col."""

    def apply(self, model):
        return _move_scalar_op_past_im2col(model, "Add")


def _move_scalar_op_past_maxpoolnhwc(model: ModelWrapper, op_type: str):
    """Move a scalar Mul/Add past a directly-following MaxPoolNHWC node.
    MaxPool commutes with any translation (max(x)+b == max(x+b)) and with
    scaling by a *positive* factor (a*max(x) == max(a*x) for a>0), so both
    the observed Mul(positive scale) and Add(bias) in this network's
    initial-block downsampling path can be pushed past MaxPoolNHWC with the
    SAME constant on both sides -- MaxPoolNHWC never combines multiple
    channels/positions with different constants, only picks a max."""
    graph = model.graph
    graph_modified = False
    for n in list(graph.node):
        if n.op_type != op_type:
            continue
        if model.is_fork_node(n) or model.is_join_node(n):
            continue
        consumer = model.find_consumer(n.output[0])
        if consumer is None or consumer.op_type != "MaxPoolNHWC" or model.is_join_node(consumer):
            continue
        A = model.get_initializer(n.input[1])
        if A is None or np.prod(A.shape) != 1:
            continue
        if op_type == "Mul" and float(np.asarray(A).reshape(-1)[0]) <= 0:
            continue  # only a positive scale commutes with max
        start_name = n.input[0]
        scalar_name = n.input[1]
        middle_name = n.output[0]
        end_name = consumer.output[0]
        attrs = {a.name: get_attribute_value(a) for a in consumer.attribute}
        new_maxpool = oh.make_node(
            "MaxPoolNHWC", [start_name], [middle_name], name=consumer.name, domain=consumer.domain, **attrs
        )
        new_scalar_op = oh.make_node(op_type, [middle_name, scalar_name], [end_name], name=n.name)
        node_ind = list(graph.node).index(n)
        graph.node.remove(n)
        graph.node.remove(consumer)
        graph.node.insert(node_ind, new_maxpool)
        graph.node.insert(node_ind + 1, new_scalar_op)
        graph_modified = True
    if graph_modified:
        model = model.transform(InferShapes())
    return model, graph_modified


class MoveScalarMulPastMaxPoolNHWC(Transformation):
    """See _move_scalar_op_past_maxpoolnhwc."""

    def apply(self, model):
        return _move_scalar_op_past_maxpoolnhwc(model, "Mul")


class MoveScalarAddPastMaxPoolNHWC(Transformation):
    """See _move_scalar_op_past_maxpoolnhwc."""

    def apply(self, model):
        return _move_scalar_op_past_maxpoolnhwc(model, "Add")


def step_absorb_leftover_scale_before_matmul(model: ModelWrapper, cfg: DataflowBuildConfig = None) -> ModelWrapper:
    """Fixup for a DecomposedPReLUAct-specific streamlining gap.

    step_enet_streamline's 4-iteration loop runs MoveScalarMulPastMatMul /
    MoveScalarAddPastMatMul (matching MatMul nodes) *before*
    LowerConvsToMatMul ever creates any MatMul node (Conv->MatMul lowering
    happens once, after the loop) -- so those two transforms never fire on
    anything. This is harmless for finn_enet_build.py's plain-QuantReLU
    configs (E1/O2/O8): their Quant->MultiThreshold dequant Mul/Add always
    ends up positioned right after a MultiThreshold with no other consumer,
    so MoveScalarMulPastConv/AbsorbMulIntoMultiThreshold fully absorb it
    during the loop, before lowering, leaving nothing for
    MoveScalarMulPastMatMul to do anyway.

    DecomposedPReLUAct's residual-adjacent fork (DuplicateStreams, shared
    between the "raw int8 for AddStreams" branch and the "dequantized float
    for the next conv" branch) blocks that in-loop absorption: Absorb*IntoMultiThreshold
    requires the MultiThreshold to be the Mul/Add's *sole* producer chain
    with no fork in between, so a scalar dequant Mul(scale)+Add(zero_point)
    pair is left stranded directly in front of the next Conv -- survives
    LowerConvsToMatMul unchanged -- ends up directly in front of an Im2Col,
    which then fails to convert (FLOAT32 input) in step_enet_convert_to_hw's
    InferConvInpGen, which in turn leaves multiple disjoint non-HW islands
    that break step_create_dataflow_partition ("cycle-free graph violated").

    Fix: re-run a MoveScalarLinearPastInvariants (past Transpose) +
    Move*PastIm2Col + Move*PastMatMul + Collapse + Absorb*IntoMultiThreshold
    sequence a few times *after* MatMul nodes actually exist (i.e. after
    step_enet_streamline, which already ran LowerConvsToMatMul), so the
    leftover scale/bias gets pushed past the intervening Transpose
    (observed: convert_to_hw-adjacent NHWC/NCHW Transposes already sit
    between the dequant Add/Mul and the Im2Col even at this pre-convert_to_hw
    stage), then past the Im2Col (verified this is where it's actually
    stranded -- MoveScalarMulPastMatMul/MoveScalarAddPastMatMul only match a
    Mul/Add whose DIRECT consumer is a MatMul, not a Transpose or Im2Col),
    and then further past the MatMul to wherever the next MultiThreshold is,
    where it's finally folded away.

    Two more gaps found once the above was working: (a) after
    MoveScalarAddPastMatMul applies its dot-product correction, the bias
    becomes a per-output-channel *vector* (no longer scalar) -- it then
    gets permanently stuck in front of the NHWC/NCHW-layout Transpose that
    sits between MatMul and the next MultiThreshold, since
    MoveScalarLinearPastInvariants only moves *scalar* constants past
    Transpose. Fix: AbsorbTransposeIntoMultiThreshold (an existing FINN
    transform for the `[0,3,1,2]`-perm case) moves the Transpose past the
    MultiThreshold instead of moving the bias past the Transpose -- either
    direction reaches the same fixed point, and Absorb*IntoMultiThreshold
    already supports 1D per-channel vectors, not just scalars. (b) the
    initial block's downsampling path has a scalar Mul(positive scale) +
    Add(bias) pair stranded directly in front of a MaxPoolNHWC node (same
    category of issue as Im2Col above, just a different blocking op) --
    fixed with the analogous MoveScalarMulPastMaxPoolNHWC/
    MoveScalarAddPastMaxPoolNHWC pair (safe since MaxPool commutes with any
    translation and with scaling by a positive factor).
    """
    for _ in range(4):
        for trn in [
            MoveScalarLinearPastInvariants(),
            MoveScalarMulPastIm2Col(),
            MoveScalarAddPastIm2Col(),
            MoveScalarMulPastMaxPoolNHWC(),
            MoveScalarAddPastMaxPoolNHWC(),
            MoveScalarMulPastMatMul(),
            MoveScalarAddPastMatMul(),
            CollapseRepeatedMul(),
            CollapseRepeatedAdd(),
            AbsorbTransposeIntoMultiThreshold(),
            AbsorbAddIntoMultiThreshold(),
            AbsorbMulIntoMultiThreshold(),
        ]:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
    return model


def _walk_back_through_transpose_and_affine(model: ModelWrapper, tensor_name: str, max_hops: int = 8):
    """Like _walk_back_affine_to_multithreshold, but also transparently
    skips over Transpose nodes (pure permutation, doesn't affect the
    scale/bias accumulation) on the way back to a MultiThreshold, and --
    critically -- the MultiThreshold found this way is ALLOWED to be a fork
    node (unlike the plain-affine walk, which requires a sole, unforked
    producer chain). That's exactly the case this exists for: an
    un-absorbable dequant chain stranded on one branch of a fork, where
    step_absorb_leftover_scale_before_matmul's forward-moving transforms
    can't finish the job because the *next* op is a hard blocker like
    FMPadding_Pixel (which always pads with a literal 0, so a trailing
    Add's bias can't be losslessly moved past it -- only Mul's scale can,
    since 0*scale==0 exactly regardless of sign). Returns
    `(multithreshold_node, scale, bias, transpose_nodes_encountered,
    affine_chain_nodes)` (transposes/chain_nodes ordered closest-to-mt
    first) or None if no such chain is found within max_hops."""
    chain = []
    transposes = []
    cur = tensor_name
    for _ in range(max_hops):
        producer = model.find_producer(cur)
        if producer is None:
            return None
        if producer.op_type == "MultiThreshold":
            scale, bias = 1.0, 0.0
            for op_type, a_val, _n in reversed(chain):
                if op_type == "Add":
                    bias = bias + a_val
                else:
                    scale = scale * a_val
                    bias = bias * a_val
            return producer, scale, bias, transposes, [c[2] for c in chain]
        if producer.op_type == "Transpose":
            if model.is_join_node(producer):
                return None
            transposes.append(producer)
            cur = producer.input[0]
            continue
        if producer.op_type not in ("Add", "Mul"):
            return None
        if model.is_fork_node(producer) or model.is_join_node(producer):
            return None
        const = model.get_initializer(producer.input[1])
        data_input = producer.input[0]
        if const is None:
            const = model.get_initializer(producer.input[0])
            data_input = producer.input[1]
        if const is None or np.asarray(const).size != 1:
            return None
        chain.append((producer.op_type, float(np.asarray(const).reshape(-1)[0]), producer))
        cur = data_input
    return None


def step_fuse_forked_dequant_into_duplicate_threshold(model: ModelWrapper, cfg: DataflowBuildConfig = None) -> ModelWrapper:
    """Last-resort fixup for a scalar Mul/Add dequant chain (optionally with
    Transposes mixed in) still stranded directly in front of a non-HW-
    convertible op (FMPadding_Pixel, MaxPoolNHWC, Im2Col) after
    step_absorb_leftover_scale_before_matmul's forward-moving loop, because
    it sits on one branch of a forked MultiThreshold (the other branch,
    e.g. the residual AddStreams input, needs the raw/differently-scaled
    value -- exactly why Absorb*IntoMultiThreshold's "sole producer" check
    refused to fold the dequant directly into the shared MultiThreshold).

    Root cause found by concrete example: a ConvTranspose's zero-insertion
    FMPadding_Pixel always pads with a literal 0 (hlslib limitation, see
    finn/src/finn/custom_op/fpgadataflow/fmpadding_pixel.py), so moving a
    trailing Add's *bias* forward past it (reusing the same constant on
    both sides, the way MoveScalarAddPastIm2Col/MaxPoolNHWC safely do) is
    NOT equivalent when bias != 0 -- the zero-gaps FMPadding_Pixel inserts
    (needed so the subsequent Im2Col+MatMul correctly reconstructs the
    strided deconvolution) would silently become `bias` instead of 0,
    changing the actual numeric result. (Mul's scale alone would be safe,
    0*scale==0 exactly, but Add is not.)

    Fix (CORRECTED -- the first version of this function baked the extra
    scale directly into the duplicate's out_scale, which is ILLEGAL: FINN's
    InferThresholdingLayer hard-asserts `out_scale == 1.0` on every
    MultiThreshold before HLS conversion -- a real, non-unit float scale
    can never be represented that way, since a Thresholding HW layer's
    output only ever steps by exactly 1 count per threshold crossing, never
    by a fractional amount. `out_bias` MAY be any *integer*, though.

    So instead: algebraically reorder the trailing `v*scale + bias` affine
    as `(v + bias/scale)*scale` (exact whenever scale != 0) -- this moves
    the *integer* part of the correction (`bias/scale`, which is exactly
    -128 in the concrete case this was written for: scale=1/128,
    bias=-1.0) onto the duplicate MultiThreshold's out_bias (legal, keeps
    out_scale == old_scale, normally 1), and pushes the *scale* part
    (`* scale`) to occur AFTER the blocking op instead of before it. This
    is safe because FMPadding_Pixel/Im2Col only ever copy or zero-fill
    values -- scaling commutes exactly with that for ANY scale (positive
    or negative), unlike translation (`+bias`), which only commutes when
    bias==0. (MaxPoolNHWC is the one exception -- max() only commutes with
    POSITIVE scale -- but that case is already handled earlier by the
    dedicated MoveScalarMulPastMaxPoolNHWC transform in
    step_absorb_leftover_scale_before_matmul's loop, which runs before this
    step; this function only ever reaches MaxPoolNHWC here if that pass
    already failed to clear it, so we conservatively require scale > 0 for
    that op type too.) The freshly-relocated Mul is left for
    step_absorb_leftover_scale_before_matmul's Move*/Collapse*/Absorb*
    transforms (re-run at the end of this function) to continue pushing
    forward and eventually absorb for good -- exactly like any other
    ordinary (non-fork-related) leftover scale.
    """
    graph = model.graph
    fused = 0
    blocking_ops = ("FMPadding_Pixel", "Im2Col", "MaxPoolNHWC")
    for node in list(graph.node):
        if node.op_type not in ("Mul", "Add"):
            continue
        if model.is_fork_node(node) or model.is_join_node(node):
            continue
        # only start a walk from the LAST node of the affine chain (i.e.
        # one whose direct consumer is a known non-HW-convertible blocking
        # op) -- processing an earlier node in the same chain independently
        # loses track of later nodes' constants (e.g. fusing a Mul but
        # silently dropping a downstream Add's bias).
        consumer = model.find_consumer(node.output[0])
        if consumer is None or consumer.op_type not in blocking_ops:
            continue
        walk = _walk_back_through_transpose_and_affine(model, node.output[0])
        if walk is None:
            continue
        mt, scale, bias, transposes, chain_nodes = walk
        if not model.is_fork_node(mt):
            continue  # non-fork cases are already handled by Absorb*IntoMultiThreshold/the Move* transforms
        if consumer.op_type == "MaxPoolNHWC" and scale <= 0:
            continue  # max() only commutes with a positive scale

        mt_inst = getCustomOp(mt)
        old_scale = mt_inst.get_nodeattr("out_scale")
        old_bias = mt_inst.get_nodeattr("out_bias")
        if old_scale != 1.0:
            continue  # shouldn't happen (mt already needs out_scale==1 for its OTHER fork branch), but be safe

        shifted_bias = bias / scale
        rounded = round(shifted_bias)
        if abs(shifted_bias - rounded) > 1e-3:
            continue  # bias/scale isn't (close to) an integer -- can't legally fold, leave for manual handling
        new_bias = old_bias + rounded  # out_scale stays == old_scale (1.0) -- LEGAL for HLS conversion

        new_mt_out = model.make_new_valueinfo_name()
        new_mt = oh.make_node(
            "MultiThreshold",
            [mt.input[0], mt.input[1]],
            [new_mt_out],
            domain=mt.domain,
            out_dtype=mt_inst.get_nodeattr("out_dtype"),
            out_scale=old_scale,
            out_bias=float(new_bias),
            data_layout=mt_inst.get_nodeattr("data_layout"),
        )

        # the last (closest-to-mt) transpose in the walk directly consumed
        # mt's original output -- repoint it to the new duplicate instead
        final_consumer = consumer
        if transposes:
            entry_transpose = transposes[-1]  # closest to mt
            entry_transpose.input[0] = new_mt_out
            exit_tensor = transposes[0].output[0]  # closest to node/final_consumer
        else:
            exit_tensor = new_mt_out

        for i, inp in enumerate(final_consumer.input):
            if inp == node.output[0]:
                final_consumer.input[i] = exit_tensor

        # splice a new Mul(scale) node in AFTER final_consumer, reusing its
        # original output tensor name so every existing downstream
        # consumer stays correctly wired without further rewiring
        old_final_out = final_consumer.output[0]
        new_intermediate = model.make_new_valueinfo_name()
        final_consumer.output[0] = new_intermediate

        # reuse an original Mul's constant array/shape if exactly one
        # contributed the whole scale (our concrete case); else fabricate
        # a plain scalar initializer.
        # NOTE: chain_nodes already includes `node` itself -- the walk in
        # _walk_back_through_transpose_and_affine starts at node.output[0],
        # whose producer IS node, so it's the first entry appended to the
        # chain. Do not append it again here (would double-list it).
        mul_srcs = [n2 for n2 in chain_nodes if n2.op_type == "Mul"]
        if len(mul_srcs) == 1:
            scale_const = model.get_initializer(mul_srcs[0].input[1])
            if scale_const is None:
                scale_const = np.asarray(scale, dtype=np.float32)
        else:
            scale_const = np.asarray(scale, dtype=np.float32)
        scale_name = model.make_new_valueinfo_name()
        model.set_initializer(scale_name, scale_const)

        new_mul = oh.make_node("Mul", [new_intermediate, scale_name], [old_final_out])

        # chain_nodes already includes `node` (see note above) -- remove each
        # exactly once.
        for n2 in chain_nodes:
            graph.node.remove(n2)
        graph.node.append(new_mt)
        graph.node.append(new_mul)
        fused += 1

    if fused:
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # the newly-spliced-in Mul needs another pass of the ordinary
        # forward-moving/absorption loop to fully resolve (it's now
        # positioned exactly like any other non-fork-related leftover
        # scale, e.g. right before an Im2Col/MatMul)
        model = step_absorb_leftover_scale_before_matmul(model, cfg)
    print(f"    [fuse_forked_dequant] fused {fused} chain(s)")
    return model


enet_estimate_steps = [
    "step_qonnx_to_finn",
    step_enet_tidy,
    step_fuse_leaky_relu_to_threshold,  # <-- fuses PReLU's LeakyRelu into a single MultiThreshold
    step_enet_streamline,
    step_absorb_leftover_scale_before_matmul,  # <-- fixes leftover dequant Mul/Add before Im2Col
    step_fuse_forked_dequant_into_duplicate_threshold,  # <-- last-resort fix for FMPadding_Pixel-adjacent leftovers
    _fixup_degenerate_signed_bias,   # <-- the only other difference vs. finn_enet_build.py
    step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

cfg_estimates = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = 1000,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xczu7ev-ffvc1156-2-e",
    steps               = enet_estimate_steps,
    generate_outputs    = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    save_intermediate_models = True,
)

if __name__ == "__main__":
    print(f"Model : {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Steps : {[s if isinstance(s, str) else s.__name__ for s in enet_estimate_steps]}")
    print()

    build.build_dataflow_cfg(MODEL_FILE, cfg_estimates)

    report_dir = os.path.join(OUTPUT_DIR, "report")
    perf_json = os.path.join(report_dir, "estimate_network_performance.json")
    res_json = os.path.join(report_dir, "estimate_layer_resources.json")

    import json

    if os.path.exists(perf_json):
        print("\n" + "=" * 60)
        print("NETWORK PERFORMANCE ESTIMATES")
        print("=" * 60)
        with open(perf_json) as f:
            perf = json.load(f)
        for k, v in perf.items():
            print(f"  {k}: {v}")

    if os.path.exists(res_json):
        print("\n" + "=" * 60)
        print("LAYER RESOURCE ESTIMATES")
        print("=" * 60)
        with open(res_json) as f:
            res = json.load(f)
        for k, v in res.items():
            print(f"  {k}: {v}")
