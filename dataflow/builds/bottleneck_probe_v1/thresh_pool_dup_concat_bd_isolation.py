"""Next bisection step after thresh_pool_bd_isolation.py (which REFUTED the "BD/stitching
mechanism" hypothesis -- see FINN_AGENT_FINAL_REPORT.md and /memories/repo/finn_gotchas.md).
Grows that minimal build by forking the thresholded stream into a skip connection: the SAME
Thresholding output feeds two parallel MaxPool(2x2,s2) branches (each still lowered to Pool_hls,
same structural params as the real probe), which are then rejoined with a channel-axis Concat --
mirroring the real initial block's "pool branch + conv branch, concatenated on channel axis"
topology, but with both branches being pools to stay minimal.

Graph: global_in -(MultiThreshold)-> thresh_out --+--(MaxPoolA 2x2,s2)--> poolA_out --+
                                                    +--(MaxPoolB 2x2,s2)--> poolB_out --+--(Concat axis=1)--> global_out

thresh_out feeding 2 consumers is a genuine fork -- to_hw.InferDuplicateStreamsLayer detects this
itself and inserts a DuplicateStreams node during to_hw conversion (not manually placed here,
same as production). The Concat starts out axis=1 (channel axis, NCHW) because each pool
branch's to_hw.InferPool lowering ends in a NHWC->NCHW Transpose(perm=[0,3,1,2]) -- but
to_hw.InferConcatLayer only lowers Concats at axis=-1 (its StreamingConcat HW op only supports
channel-last concatenation), so _move_transpose_past_concat (copied from
hardware/finn_enet_build.py's MoveTransposePastJoinConcat, trimmed, no other repo deps) pushes
both trailing Transposes past the Concat first, rewriting it to axis=-1 with a single trailing
Transpose back to NCHW -- exactly the real initial block's Concat-prep trick.

Run (inside the FINN container):
    docker cp thresh_pool_dup_concat_bd_isolation.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir <container> bash -c \
        'source /tools/Xilinx/Vivado/2022.2/settings64.sh && \
         cd /home/thelegendiv/finn/notebooks/enet && python3 thresh_pool_dup_concat_bd_isolation.py'
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
CHANNELS = 1

out_dir = os.path.join(
    ENET_DIR, "finn_deployment_outputs", f"thresh_pool_dup_concat_bd_isolation_{time.strftime('%Y%m%d_%H%M%S')}"
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
    """global_in (UINT8, NCHW 1x1x256x256) -> MultiThreshold (-> UINT4) -> fork into two
    MaxPool(2x2,s2) branches -> Concat(axis=1) -> global_out (UINT4, 1x2x128x128)."""
    in_shape = [1, CHANNELS, IFM_DIM, IFM_DIM]
    pool_shape = [1, CHANNELS, IFM_DIM // 2, IFM_DIM // 2]
    out_shape = [1, CHANNELS * 2, IFM_DIM // 2, IFM_DIM // 2]
    n_thresh = 15  # UINT4 output: 16 levels -> 15 thresholds
    thresholds = np.sort(np.random.randint(1, 255, size=(CHANNELS, n_thresh)).astype(np.float32), axis=1)

    global_in = oh.make_tensor_value_info("global_in", TensorProto.FLOAT, in_shape)
    global_out = oh.make_tensor_value_info("global_out", TensorProto.FLOAT, out_shape)
    thresh_out = oh.make_tensor_value_info("thresh_out", TensorProto.FLOAT, in_shape)
    poolA_out = oh.make_tensor_value_info("poolA_out", TensorProto.FLOAT, pool_shape)
    poolB_out = oh.make_tensor_value_info("poolB_out", TensorProto.FLOAT, pool_shape)
    thresh_init = oh.make_tensor("thresholds", TensorProto.FLOAT, thresholds.shape, thresholds.flatten())

    thresh_node = oh.make_node(
        "MultiThreshold", ["global_in", "thresholds"], ["thresh_out"],
        domain="qonnx.custom_op.general", out_dtype="UINT4", out_scale=1.0, out_bias=0.0, data_layout="NCHW",
    )
    # thresh_out feeds BOTH pool nodes below -> a genuine fork. to_hw.InferDuplicateStreamsLayer
    # detects this itself later and inserts a DuplicateStreams node (not manually placed here,
    # same mechanism as the real initial block's skip connection).
    poolA_node = oh.make_node(
        "MaxPool", ["thresh_out"], ["poolA_out"], name="MaxPoolA", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )
    poolB_node = oh.make_node(
        "MaxPool", ["thresh_out"], ["poolB_out"], name="MaxPoolB", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )
    concat_node = oh.make_node("Concat", ["poolA_out", "poolB_out"], ["global_out"], axis=1)

    graph = oh.make_graph(
        [thresh_node, poolA_node, poolB_node, concat_node], "thresh_pool_dup_concat_isolation",
        [global_in], [global_out], value_info=[thresh_out, poolA_out, poolB_out], initializer=[thresh_init],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresh_pool_dup_concat_bd_isolation"))
    model.set_tensor_datatype("global_in", DataType["UINT8"])
    model.set_tensor_datatype("thresholds", DataType["UINT8"])
    model.set_tensor_datatype("thresh_out", DataType["UINT4"])
    model.set_tensor_datatype("poolA_out", DataType["UINT4"])
    model.set_tensor_datatype("poolB_out", DataType["UINT4"])
    model.set_tensor_datatype("global_out", DataType["UINT4"])
    return model


def main() -> None:
    m = build_model()
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(InferDataLayouts())
    m = m.transform(GiveUniqueNodeNames())
    m.save(os.path.join(out_dir, "step00_initial.onnx"))

    # same InferPool monkeypatch as thresh_pool_bd_isolation.py / finn_bottleneck_probe_build.py's
    # pool_impl=="pool" route -- keeps both branches lowering to Pool_hls (not the legacy
    # StreamingMaxPool_hls op), matching the real crashing probe's op type.
    to_hw.InferStreamingMaxPool = getattr(to_hw, "InferPool", None) or getattr(to_hw, "InferPool_Batch")
    print("[isolation] pool lowered with", to_hw.InferStreamingMaxPool.__name__, flush=True)

    for trn in [
        to_hw.InferStreamingMaxPool,  # MaxPool (NCHW) -> Transpose + Im2Col(depthwise) + Pool + Transpose, x2 branches
        RoundAndClipThresholds,
        to_hw.InferThresholdingLayer,  # MultiThreshold -> standalone Thresholding
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,  # Im2Col -> ConvolutionInputGenerator
        to_hw.InferDuplicateStreamsLayer,  # Thresholding's 2-consumer fork -> DuplicateStreams
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
