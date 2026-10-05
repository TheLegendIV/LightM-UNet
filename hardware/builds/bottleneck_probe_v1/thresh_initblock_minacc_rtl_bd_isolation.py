"""Accumulator-width retest, 3rd hypothesis after thresh_initblock_narrowweights_rtl_bd_isolation.py
(narrow_weights: REFUTED, identical crash) and thresh_initblock_simd9_rtl_bd_isolation.py
(SIMD=1 vs SIMD=9 matching the real probe's landed folding: ALSO REFUTED, byte-for-byte
identical crash at mvu_4sx4u.sv:556/513/133 both times).

Dumping this family's MVAU_rtl nodeattrs after specialize found: accDataType = outputDataType =
**INT32** -- our isolation scripts never run FINN's MinimizeAccumulatorWidth pass (production's
step_minimize_bit_width does), so the accumulator sits at a generic unminimized width instead of
the realistic ~11-12 bits needed for MW=9/UINT4(act)xINT4(narrow weight) (max sum
9*15*7=945). The failing synthesis error is `part-select [54:23] out of range of prefix 'p3'`
-- 54-23+1 = **32 bits**, suspiciously matching INT32 exactly. This script is
thresh_initblock_narrowweights_rtl_bd_isolation.py (narrow weights, SIMD left at default 1/1 --
proven irrelevant by the sibling test, no need to force 9 anymore) PLUS
`finn.transformation.fpgadataflow.minimize_accumulator_width.MinimizeAccumulatorWidth()` run
right after `to_hw.InferQuantizedMatrixVectorActivation`, narrowing accDataType/outputDataType to
their real minimal width before specialize/codegen -- to test whether the artificially-wide INT32
accumulator (not weight data, not folding) is what trips the mvu_4sx4u.sv part-select bug.

Run (inside the FINN container):
    docker cp thresh_initblock_minacc_rtl_bd_isolation.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir <container> bash -c \
        'source /tools/Xilinx/Vivado/2022.2/settings64.sh && \
         cd /home/thelegendiv/finn/notebooks/enet && python3 thresh_initblock_minacc_rtl_bd_isolation.py'
"""
import os
import sys
import time

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
sys.path.insert(0, ENET_DIR)

_XILINX_BIN_DIRS = ["/tools/Xilinx/Vitis_HLS/2022.2/bin", "/tools/Xilinx/Vivado/2022.2/bin"]
os.environ["PATH"] = os.pathsep.join(_XILINX_BIN_DIRS + [os.environ.get("PATH", "")])
os.environ.setdefault("XILINX_VIVADO", "/tools/Xilinx/Vivado/2022.2")
os.environ.setdefault("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2022.2")

FPGA_PART = "xczu7ev-ffvc1156-2-e"
CLK_NS = 10.0
IFM_DIM = 256
CIN = 1
COUT = 4  # matches the real init_cin1_cout4_in256_int4 probe naming

out_dir = os.path.join(
    ENET_DIR, "finn_deployment_outputs", f"thresh_initblock_minacc_rtl_bd_isolation_{time.strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(os.path.join(out_dir, "report"), exist_ok=True)
os.environ.setdefault("FINN_BUILD_DIR", os.path.join(out_dir, "finn_build_tmp"))
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
print(f"out  {out_dir}", flush=True)

import numpy as np  # noqa: E402
import onnx.helper as oh  # noqa: E402
from onnx import TensorProto  # noqa: E402

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.base import Transformation  # noqa: E402
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames  # noqa: E402
from qonnx.transformation.infer_data_layouts import InferDataLayouts  # noqa: E402
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: E402
from qonnx.transformation.infer_shapes import InferShapes  # noqa: E402
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.util.basic import get_by_name, qonnx_make_model  # noqa: E402

import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
from finn.builder.build_dataflow_config import DataflowBuildConfig  # noqa: E402
from finn.builder.build_dataflow_steps import (  # noqa: E402
    step_create_dataflow_partition,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_specialize_layers,
)
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw  # noqa: E402
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth  # noqa: E402
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes  # noqa: E402
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds  # noqa: E402
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs, reset_implementation  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402


def _move_transpose_past_concat(model: ModelWrapper, perm=(0, 3, 1, 2)):
    """Trimmed copy of hardware/finn_enet_build.py's free function of the same name (kept
    self-contained here on purpose)."""
    graph = model.graph
    graph_modified = False
    for concat_node in list(graph.node):
        if concat_node.op_type != "Concat":
            continue
        axis_attr = get_by_name(concat_node.attribute, "axis")
        if axis_attr is None or axis_attr.i != 1:
            continue
        producers = [model.find_producer(t) for t in concat_node.input]
        if any(p is None or p.op_type != "Transpose" for p in producers):
            continue
        if any(model.is_fork_node(p) or model.is_join_node(p) for p in producers):
            continue
        perms = [get_by_name(p.attribute, "perm") for p in producers]
        if any(pm is None or list(pm.ints) != list(perm) for pm in perms):
            continue
        data_inputs = [p.input[0] for p in producers]
        ishapes = [model.get_tensor_shape(x) for x in data_inputs]
        if any(s is None or len(s) != 4 for s in ishapes):
            continue
        end_name = concat_node.output[0]
        middle_name = end_name + "_pre_transpose"
        correct_out_shape = model.get_tensor_shape(end_name)  # NCHW, already correct
        new_concat = oh.make_node(
            "Concat", data_inputs, [middle_name], name=concat_node.name, domain=concat_node.domain, axis=-1
        )
        new_transpose = oh.make_node("Transpose", [middle_name], [end_name], name=producers[0].name, perm=list(perm))
        node_ind = list(graph.node).index(concat_node)
        for p in producers:
            graph.node.remove(p)
        graph.node.remove(concat_node)
        graph.node.insert(node_ind, new_concat)
        graph.node.insert(node_ind + 1, new_transpose)
        n0, c0, h0, w0 = correct_out_shape
        model.set_tensor_shape(middle_name, [n0, h0, w0, c0])  # pre-transpose NHWC shape
        graph_modified = True
    if graph_modified:
        model = model.transform(InferShapes())
    return model, graph_modified


class MoveTransposePastJoinConcat(Transformation):
    """See _move_transpose_past_concat."""

    def apply(self, model):
        return _move_transpose_past_concat(model)


def build_model() -> ModelWrapper:
    """Identical topology to thresh_initblock_bd_isolation.py -- see that file's docstring.
    Keeps the true-narrow-range [-7,7] weights (proven neutral, no harm)."""
    cout_conv = COUT - CIN  # 3
    in_shape = [1, CIN, IFM_DIM, IFM_DIM]
    pool_shape = [1, CIN, IFM_DIM // 2, IFM_DIM // 2]
    conv_shape = [1, cout_conv, IFM_DIM // 2, IFM_DIM // 2]
    out_shape = [1, COUT, IFM_DIM // 2, IFM_DIM // 2]
    n_thresh = 15  # UINT4 output: 16 levels -> 15 thresholds

    thresh0 = np.sort(np.random.randint(1, 255, size=(CIN, n_thresh)).astype(np.float32), axis=1)
    thresh1 = np.sort(np.random.randint(-2000, 2000, size=(cout_conv, n_thresh)).astype(np.float32), axis=1)
    conv_w = np.random.randint(-7, 8, size=(cout_conv, CIN, 3, 3)).astype(np.float32)  # true narrow INT4

    global_in = oh.make_tensor_value_info("global_in", TensorProto.FLOAT, in_shape)
    global_out = oh.make_tensor_value_info("global_out", TensorProto.FLOAT, out_shape)
    thresh_out = oh.make_tensor_value_info("thresh_out", TensorProto.FLOAT, in_shape)
    poolA_out = oh.make_tensor_value_info("poolA_out", TensorProto.FLOAT, pool_shape)
    convB_raw = oh.make_tensor_value_info("convB_raw", TensorProto.FLOAT, conv_shape)
    convB_out = oh.make_tensor_value_info("convB_out", TensorProto.FLOAT, conv_shape)
    thresh0_init = oh.make_tensor("thresholds0", TensorProto.FLOAT, thresh0.shape, thresh0.flatten())
    thresh1_init = oh.make_tensor("thresholds1", TensorProto.FLOAT, thresh1.shape, thresh1.flatten())
    conv_w_init = oh.make_tensor("conv_weight", TensorProto.FLOAT, conv_w.shape, conv_w.flatten())

    thresh_node = oh.make_node(
        "MultiThreshold", ["global_in", "thresholds0"], ["thresh_out"],
        domain="qonnx.custom_op.general", out_dtype="UINT4", out_scale=1.0, out_bias=0.0, data_layout="NCHW",
    )
    pool_node = oh.make_node(
        "MaxPool", ["thresh_out"], ["poolA_out"], name="MaxPoolA", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )
    conv_node = oh.make_node(
        "Conv", ["thresh_out", "conv_weight"], ["convB_raw"], name="ConvB",
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], group=1,
    )
    thresh2_node = oh.make_node(
        "MultiThreshold", ["convB_raw", "thresholds1"], ["convB_out"],
        domain="qonnx.custom_op.general", out_dtype="UINT4", out_scale=1.0, out_bias=0.0, data_layout="NCHW",
    )
    concat_node = oh.make_node("Concat", ["poolA_out", "convB_out"], ["global_out"], axis=1)

    graph = oh.make_graph(
        [thresh_node, pool_node, conv_node, thresh2_node, concat_node], "thresh_initblock_minacc_isolation",
        [global_in], [global_out], value_info=[thresh_out, poolA_out, convB_raw, convB_out],
        initializer=[thresh0_init, thresh1_init, conv_w_init],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresh_initblock_minacc_rtl_bd_isolation"))
    model.set_tensor_datatype("global_in", DataType["UINT8"])
    model.set_tensor_datatype("thresholds0", DataType["UINT8"])
    model.set_tensor_datatype("thresh_out", DataType["UINT4"])
    model.set_tensor_datatype("poolA_out", DataType["UINT4"])
    model.set_tensor_datatype("conv_weight", DataType["INT4"])
    model.set_tensor_datatype("convB_raw", DataType["INT16"])  # raw conv accumulator, pre-threshold
    model.set_tensor_datatype("thresholds1", DataType["INT16"])
    model.set_tensor_datatype("convB_out", DataType["UINT4"])
    model.set_tensor_datatype("global_out", DataType["UINT4"])
    return model


def main() -> None:
    m = build_model()
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(InferDataLayouts())
    m = m.transform(GiveUniqueNodeNames())
    m.save(os.path.join(out_dir, "step00_initial.onnx"))

    m = m.transform(LowerConvsToMatMul())
    m = m.transform(InferDataTypes())
    m = m.transform(InferDataLayouts())
    m = m.transform(GiveUniqueNodeNames())
    print("[isolation] after LowerConvsToMatMul ops:", sorted({n.op_type for n in m.graph.node}), flush=True)

    to_hw.InferStreamingMaxPool = getattr(to_hw, "InferPool", None) or getattr(to_hw, "InferPool_Batch")
    print("[isolation] pool lowered with", to_hw.InferStreamingMaxPool.__name__, flush=True)

    for trn in [
        to_hw.InferStreamingMaxPool,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
    ]:
        m = m.transform(trn())
        m = m.transform(InferDataLayouts())
        m = m.transform(GiveUniqueNodeNames())
        m = m.transform(InferDataTypes())
    print("[isolation] after first to_hw pass ops:", sorted({n.op_type for n in m.graph.node}), flush=True)

    for n in m.graph.node:
        if n.op_type == "MVAU":
            op = getCustomOp(n)
            print(f"[isolation] {n.name} accDataType BEFORE minimize: {op.get_nodeattr('accDataType')}", flush=True)
    m = m.transform(MinimizeAccumulatorWidth())
    m = m.transform(InferDataTypes())
    for n in m.graph.node:
        if n.op_type == "MVAU":
            op = getCustomOp(n)
            print(f"[isolation] {n.name} accDataType AFTER minimize: {op.get_nodeattr('accDataType')}", flush=True)

    m, moved = _move_transpose_past_concat(m)
    print("[isolation] moved transpose past concat:", moved, flush=True)
    m = m.transform(InferDataTypes())
    for trn in [to_hw.InferConcatLayer, AbsorbConsecutiveTransposes]:
        m = m.transform(trn())
        m = m.transform(InferDataLayouts())
        m = m.transform(GiveUniqueNodeNames())
        m = m.transform(InferDataTypes())
    m = m.transform(GiveReadableTensorNames())
    print("[isolation] after convert_to_hw ops:", sorted({n.op_type for n in m.graph.node}), flush=True)
    m.save(os.path.join(out_dir, "step01_after_convert_to_hw.onnx"))

    # NOTE: NOT forcing preferred_impl_style="hls" -- let MVAU auto-select RTL.

    cfg = DataflowBuildConfig(
        output_dir=out_dir, mvau_wwidth_max=80, target_fps=None, synth_clk_period_ns=CLK_NS, fpga_part=FPGA_PART,
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod("largefifo_rtlsim"),
        generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP], save_intermediate_models=True, steps=[],
    )

    m = step_create_dataflow_partition(m, cfg)
    m = step_specialize_layers(m, cfg)
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    print("[isolation] after specialize ops:", sorted({n.op_type for n in m.graph.node}), flush=True)
    for n in m.graph.node:
        if n.op_type == "MVAU_rtl":
            mop = getCustomOp(n)
            print(
                f"[isolation] {n.name}: MW={mop.get_nodeattr('MW')} MH={mop.get_nodeattr('MH')} "
                f"PE={mop.get_nodeattr('PE')} SIMD={mop.get_nodeattr('SIMD')} "
                f"accDataType={mop.get_nodeattr('accDataType')} outputDataType={mop.get_nodeattr('outputDataType')}",
                flush=True,
            )
    m.save(os.path.join(out_dir, "step02_after_specialize.onnx"))

    # Same StreamingConcat full-unfold fix as the sibling scripts.
    for cnode in m.graph.node:
        if not cnode.op_type.startswith("StreamingConcat"):
            continue
        for inp in cnode.input:
            producer = m.find_producer(inp)
            if producer is None:
                continue
            pop = getCustomOp(producer)
            attr_types = pop.get_nodeattr_types()
            if "PE" in attr_types and "NumChannels" in attr_types:
                nc = pop.get_nodeattr("NumChannels")
                if pop.get_nodeattr("PE") != nc:
                    print(f"[isolation] forcing {producer.name} PE {pop.get_nodeattr('PE')} -> {nc}", flush=True)
                    pop.set_nodeattr("PE", nc)

    m = step_hw_codegen(m, cfg)
    m = step_hw_ipgen(m, cfg)
    m = step_set_fifo_depths(m, cfg)
    m.save(os.path.join(out_dir, "step03_after_fifo_sizing.onnx"))

    m = m.transform(SplitLargeFIFOs())
    m = m.transform(GiveUniqueNodeNames())
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            reset_implementation(getCustomOp(n))
    m = m.transform(PrepareIP(FPGA_PART, CLK_NS))
    m = m.transform(HLSSynthIP())
    m.save(os.path.join(out_dir, "step04_after_forced_fifo.onnx"))

    print("[isolation] running CreateStitchedIP (real Vivado BD/IP-Integrator flow)...", flush=True)
    m = m.transform(CreateStitchedIP(FPGA_PART, CLK_NS))
    m.save(os.path.join(out_dir, "step05_stitched.onnx"))

    print("[isolation] running SynthOutOfContext (real vivadocompile.tcl, unmodified)...", flush=True)
    m = m.transform(SynthOutOfContext(part=FPGA_PART, clk_period_ns=CLK_NS))
    res = eval(m.get_metadata_prop("res_total_ooc_synth"))
    print("[isolation] OOC SYNTH RESULT:", res, flush=True)
    m.save(os.path.join(out_dir, "step06_final.onnx"))
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
