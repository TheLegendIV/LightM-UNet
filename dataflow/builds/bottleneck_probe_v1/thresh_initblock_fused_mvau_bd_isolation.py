"""Parallel variant of thresh_initblock_bd_isolation.py, run alongside it (not sequentially) to
test whether the RTL MVAU Verilog synthesis bug hit there (finn-rtllib's mvu_4sx4u.sv:
"part-select out of range"/"conditional expression could not be resolved to a constant" for
MW=9/MH=3/PE=1/SIMD=1/UINT4xINT4) is specific to noActivation=1 (standalone Thresholding after a
plain MVAU, what the other script produces) or also occurs with noActivation=0 (MultiThreshold
FUSED directly into the MVAU as its output activation, same ActVal/out_scale mechanism
InferQuantizedMatrixVectorActivation uses for regular production conv layers).

to_hw.InferQuantizedMatrixVectorActivation only fuses a MultiThreshold that is the MatMul's
*direct* consumer (mm_output's sole consumer, checked via model.find_consumer). Plain
qonnx.transformation.lower_convs_to_matmul.LowerConvsToMatMul always inserts its own trailing
NHWC->NCHW Transpose between the raw MatMul and whatever consumes the conv's declared output
name, which defeats this direct-consumer check (confirmed in thresh_initblock_bd_isolation.py --
that script produces a standalone Thresholding_rtl after MVAU_hls/MVAU_rtl, not a fused one). To
force the fused path here, this script hand-builds the conv branch ALREADY in Im2Col+MatMul
lowered form (mirroring exactly what LowerConvsToMatMul generates internally, just without its
own trailing Transpose/threshold-order issue), with the 2nd MultiThreshold placed as the MatMul's
direct consumer in NHWC space, and only adds the NHWC->NCHW Transpose AFTER that fused MVAU.

Graph: global_in -(MultiThreshold)-> thresh_out --+--(MaxPool 2x2,s2)----------------------------------------------> poolA_out(C=1) --+
                                                    +-(Transpose NCHW->NHWC)-(Im2Col 3x3,s2,p1)-(MatMul)-(MultiThreshold, FUSED)-(Transpose NHWC->NCHW)-> convB_out(C=3) --+--(Concat axis=1)--> global_out(C=4)

Deliberately does NOT force preferred_impl_style="hls" on the MVAU (unlike the sibling script) --
left at the default auto-selection (which chooses RTL for bit-widths >=4) specifically so this
run can show whether the RTL codegen bug also fires in the fused (noActivation=0) case.

Run (inside the FINN container):
    docker cp thresh_initblock_fused_mvau_bd_isolation.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir <container> bash -c \
        'source /tools/Xilinx/Vivado/2022.2/settings64.sh && \
         cd /home/thelegendiv/finn/notebooks/enet && python3 thresh_initblock_fused_mvau_bd_isolation.py'
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
    ENET_DIR, "finn_deployment_outputs", f"thresh_initblock_fused_mvau_bd_isolation_{time.strftime('%Y%m%d_%H%M%S')}"
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
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes  # noqa: E402
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds  # noqa: E402
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP  # noqa: E402
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP  # noqa: E402
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP  # noqa: E402
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs, reset_implementation  # noqa: E402
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext  # noqa: E402


def _move_transpose_past_concat(model: ModelWrapper, perm=(0, 3, 1, 2)):
    """Trimmed copy of hardware/finn_enet_build.py's free function of the same name (kept
    self-contained here on purpose -- see module docstring). Rewrites an axis=1 Concat whose
    every input is produced by an identical, non-forked Transpose(perm) into an axis=-1 Concat
    followed by a single trailing Transpose(perm)."""
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
    """global_in (UINT8, NCHW 1x1x256x256) -> MultiThreshold (-> UINT4) -> fork into
    MaxPool(2x2,s2) [pool branch, C=1] and a hand-lowered Transpose+Im2Col+MatMul+MultiThreshold
    conv branch [C=3, threshold placed as MatMul's DIRECT consumer so InferQuantizedMatrix
    VectorActivation fuses it, noActivation=0] -> Concat(axis=1) -> global_out (UINT4,
    1x4x128x128) -- same real initial-block topology as thresh_initblock_bd_isolation.py, but
    with the conv branch's activation fused into the MVAU instead of a separate Thresholding."""
    cout_conv = COUT - CIN  # 3
    kh = kw = 3
    im2col_w = CIN * kh * kw  # 9
    in_shape = [1, CIN, IFM_DIM, IFM_DIM]
    pool_shape = [1, CIN, IFM_DIM // 2, IFM_DIM // 2]
    nhwc_in_shape = [1, IFM_DIM, IFM_DIM, CIN]
    im2col_shape = [1, IFM_DIM // 2, IFM_DIM // 2, im2col_w]
    matmul_shape = [1, IFM_DIM // 2, IFM_DIM // 2, cout_conv]
    conv_shape = [1, cout_conv, IFM_DIM // 2, IFM_DIM // 2]
    out_shape = [1, COUT, IFM_DIM // 2, IFM_DIM // 2]
    n_thresh = 15  # UINT4 output: 16 levels -> 15 thresholds

    thresh0 = np.sort(np.random.randint(1, 255, size=(CIN, n_thresh)).astype(np.float32), axis=1)
    thresh1 = np.sort(np.random.randint(-2000, 2000, size=(cout_conv, n_thresh)).astype(np.float32), axis=1)
    # already in LowerConvsToMatMul's post-lowering layout: (IFM*kh*kw, OFM) = (9, 3) -- see
    # that transform's source (qonnx.transformation.lower_convs_to_matmul) for the derivation,
    # reproduced here directly since no real numerical conv equivalence is needed for this test.
    conv_w = np.random.randint(-8, 8, size=(im2col_w, cout_conv)).astype(np.float32)  # INT4 range

    global_in = oh.make_tensor_value_info("global_in", TensorProto.FLOAT, in_shape)
    global_out = oh.make_tensor_value_info("global_out", TensorProto.FLOAT, out_shape)
    thresh_out = oh.make_tensor_value_info("thresh_out", TensorProto.FLOAT, in_shape)
    poolA_out = oh.make_tensor_value_info("poolA_out", TensorProto.FLOAT, pool_shape)
    conv_in_nhwc = oh.make_tensor_value_info("conv_in_nhwc", TensorProto.FLOAT, nhwc_in_shape)
    im2col_out = oh.make_tensor_value_info("im2col_out", TensorProto.FLOAT, im2col_shape)
    matmul_out = oh.make_tensor_value_info("matmul_out", TensorProto.FLOAT, matmul_shape)
    convB_out_nhwc = oh.make_tensor_value_info("convB_out_nhwc", TensorProto.FLOAT, matmul_shape)
    convB_out = oh.make_tensor_value_info("convB_out", TensorProto.FLOAT, conv_shape)
    thresh0_init = oh.make_tensor("thresholds0", TensorProto.FLOAT, thresh0.shape, thresh0.flatten())
    thresh1_init = oh.make_tensor("thresholds1", TensorProto.FLOAT, thresh1.shape, thresh1.flatten())
    conv_w_init = oh.make_tensor("conv_weight", TensorProto.FLOAT, conv_w.shape, conv_w.flatten())

    thresh_node = oh.make_node(
        "MultiThreshold", ["global_in", "thresholds0"], ["thresh_out"],
        domain="qonnx.custom_op.general", out_dtype="UINT4", out_scale=1.0, out_bias=0.0, data_layout="NCHW",
    )
    # thresh_out feeds BOTH branches below -> a genuine fork. to_hw.InferDuplicateStreamsLayer
    # detects this itself later and inserts a DuplicateStreams node (not manually placed here,
    # same mechanism as the real initial block's skip connection).
    pool_node = oh.make_node(
        "MaxPool", ["thresh_out"], ["poolA_out"], name="MaxPoolA", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )
    # hand-lowered conv branch (mirrors LowerConvsToMatMul's own internal construction -- see
    # module docstring for why this is built by hand instead of just running that transform).
    inp_trans_node = oh.make_node("Transpose", ["thresh_out"], ["conv_in_nhwc"], name="ConvInTranspose", perm=[0, 2, 3, 1])
    im2col_node = oh.make_node(
        "Im2Col", ["conv_in_nhwc"], ["im2col_out"], name="ConvIm2Col", domain="qonnx.custom_op.general",
        stride=[2, 2], kernel_size=[kh, kw], pad_amount=[1, 1, 1, 1],
        input_shape=f"(1,{IFM_DIM},{IFM_DIM},{CIN})", depthwise=0, dilations=[1, 1],
    )
    matmul_node = oh.make_node("MatMul", ["im2col_out", "conv_weight"], ["matmul_out"], name="ConvMatMul")
    # MultiThreshold is the MatMul's DIRECT consumer (no Transpose in between) -> the one
    # structural difference from thresh_initblock_bd_isolation.py that lets
    # InferQuantizedMatrixVectorActivation fuse it into the MVAU (noActivation=0).
    thresh2_node = oh.make_node(
        "MultiThreshold", ["matmul_out", "thresholds1"], ["convB_out_nhwc"],
        domain="qonnx.custom_op.general", out_dtype="UINT4", out_scale=1.0, out_bias=0.0, data_layout="NHWC",
    )
    out_trans_node = oh.make_node("Transpose", ["convB_out_nhwc"], ["convB_out"], name="ConvOutTranspose", perm=[0, 3, 1, 2])
    concat_node = oh.make_node("Concat", ["poolA_out", "convB_out"], ["global_out"], axis=1)

    graph = oh.make_graph(
        [thresh_node, pool_node, inp_trans_node, im2col_node, matmul_node, thresh2_node, out_trans_node, concat_node],
        "thresh_initblock_fused_mvau_isolation", [global_in], [global_out],
        value_info=[thresh_out, poolA_out, conv_in_nhwc, im2col_out, matmul_out, convB_out_nhwc, convB_out],
        initializer=[thresh0_init, thresh1_init, conv_w_init],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresh_initblock_fused_mvau_bd_isolation"))
    model.set_tensor_datatype("global_in", DataType["UINT8"])
    model.set_tensor_datatype("thresholds0", DataType["UINT8"])
    model.set_tensor_datatype("thresh_out", DataType["UINT4"])
    model.set_tensor_datatype("poolA_out", DataType["UINT4"])
    model.set_tensor_datatype("conv_in_nhwc", DataType["UINT4"])
    model.set_tensor_datatype("conv_weight", DataType["INT4"])
    model.set_tensor_datatype("im2col_out", DataType["UINT4"])
    model.set_tensor_datatype("matmul_out", DataType["INT16"])  # raw conv accumulator, pre-threshold
    model.set_tensor_datatype("thresholds1", DataType["INT16"])
    model.set_tensor_datatype("convB_out_nhwc", DataType["UINT4"])
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

    # same InferPool monkeypatch as the sibling isolation scripts -- keeps the pool branch
    # lowering to Pool_hls (not the legacy StreamingMaxPool_hls op), matching the real probe.
    to_hw.InferStreamingMaxPool = getattr(to_hw, "InferPool", None) or getattr(to_hw, "InferPool_Batch")
    print("[isolation] pool lowered with", to_hw.InferStreamingMaxPool.__name__, flush=True)

    for trn in [
        to_hw.InferStreamingMaxPool,  # MaxPool (NCHW) -> Transpose + Im2Col(depthwise) + Pool + Transpose
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,  # hand-built MatMul+MultiThreshold -> single fused MVAU (noActivation=0)
        to_hw.InferThresholdingLayer,  # remaining standalone MultiThreshold (thresh_out's own) -> Thresholding
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,  # both branches' Im2Col -> ConvolutionInputGenerator
        to_hw.InferDuplicateStreamsLayer,  # thresh_out's 2-consumer fork -> DuplicateStreams
    ]:
        m = m.transform(trn())
        m = m.transform(InferDataLayouts())
        m = m.transform(GiveUniqueNodeNames())
        m = m.transform(InferDataTypes())
    print("[isolation] after first to_hw pass ops:", sorted({n.op_type for n in m.graph.node}), flush=True)

    # Concat-prep: push both branches' trailing NHWC->NCHW Transposes past the axis=1 Concat,
    # rewriting it to axis=-1 (channel-last), then lower it to StreamingConcat.
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

    # deliberately NOT forcing preferred_impl_style="hls" here (unlike the sibling script) --
    # left at default auto-selection (RTL for bit-widths >=4) to test whether the fused
    # (noActivation=0) MVAU also hits the mvu_4sx4u.sv RTL codegen bug or not.

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
    m.save(os.path.join(out_dir, "step02_after_specialize.onnx"))

    # Same fix as thresh_initblock_bd_isolation.py, generalized across nodeattr naming
    # conventions ("NumChannels" for Thresholding/Pool-style ops, "MH" for MVAU): StreamingConcat
    # assumes each branch delivers ALL its channels in a single stream element per cycle (PE ==
    # full channel count). Force full unfold on every direct producer of a StreamingConcat node
    # before codegen/ipgen run.
    for cnode in m.graph.node:
        if not cnode.op_type.startswith("StreamingConcat"):
            continue
        for inp in cnode.input:
            producer = m.find_producer(inp)
            if producer is None:
                continue
            pop = getCustomOp(producer)
            attr_types = pop.get_nodeattr_types()
            if "PE" not in attr_types:
                continue
            full = None
            if "NumChannels" in attr_types:
                full = pop.get_nodeattr("NumChannels")
            elif "MH" in attr_types:
                full = pop.get_nodeattr("MH")
            if full is None:
                continue
            if pop.get_nodeattr("PE") != full:
                print(f"[isolation] forcing {producer.name} PE {pop.get_nodeattr('PE')} -> {full}", flush=True)
                pop.set_nodeattr("PE", full)

    m = step_hw_codegen(m, cfg)
    m = step_hw_ipgen(m, cfg)
    m = step_set_fifo_depths(m, cfg)
    m.save(os.path.join(out_dir, "step03_after_fifo_sizing.onnx"))

    m = m.transform(SplitLargeFIFOs())
    m = m.transform(GiveUniqueNodeNames())
    # see finn_bottleneck_probe_build.py: SplitLargeFIFOs can renumber nodes after codegen, so force every
    # FIFO's IP to regenerate unconditionally (avoids a stale-name/fresh-name mismatch in CreateStitchedIP)
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
