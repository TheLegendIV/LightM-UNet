"""Minimal 2-op FINN build: MultiThreshold -> MaxPool, same structural params as the
init_cin1_cout4_in256_int4_pool_thrpre probe's pool branch (channels=1, 256x256 -> 128x128,
kernel/stride 2x2, INT4), run through the REAL FINN pipeline (specialize_layers -> hw_codegen
-> hw_ipgen -> set_fifo_depths -> CreateStitchedIP [real Vivado IP-Integrator/BD stitching] ->
SynthOutOfContext [real vivadocompile.tcl, unmodified]).

Purpose: distinguish whether the bottleneck_probe_v1 Pool_hls OOC-synth crash needs the real
BD/IP-Integrator stitching mechanism to reproduce (vs. just needing more surrounding IPs/scale
regardless of stitching mechanism) -- see FINN_AGENT_FINAL_REPORT.md item 3 and
/memories/repo/finn_gotchas.md. Deliberately skips ALL of this repo's enet-specific
streamlining/role-identification/folding infrastructure (finn_enet_build.py,
finn_bottleneck_probe_build.py) -- just the directed minimum of FINN transforms needed to turn
a hand-built MultiThreshold+MaxPool ONNX graph into a stitched, OOC-synthesizable IP.

Run (inside the FINN container):
    docker cp thresh_pool_bd_isolation.py <container>:/home/thelegendiv/finn/notebooks/enet/
    docker exec -e HOME=/tmp/home_dir <container> bash -c \
        'source /tools/Xilinx/Vivado/2022.2/settings64.sh && \
         cd /home/thelegendiv/finn/notebooks/enet && python3 thresh_pool_bd_isolation.py'
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

out_dir = os.path.join(ENET_DIR, "finn_deployment_outputs", f"thresh_pool_bd_isolation_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(os.path.join(out_dir, "report"), exist_ok=True)
os.environ.setdefault("FINN_BUILD_DIR", os.path.join(out_dir, "finn_build_tmp"))
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
print(f"out  {out_dir}", flush=True)

import numpy as np  # noqa: E402
import onnx.helper as oh  # noqa: E402
from onnx import TensorProto  # noqa: E402

from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames  # noqa: E402
from qonnx.transformation.infer_data_layouts import InferDataLayouts  # noqa: E402
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: E402
from qonnx.transformation.infer_shapes import InferShapes  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.util.basic import qonnx_make_model  # noqa: E402

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


def build_model() -> ModelWrapper:
    """global_in (UINT8, NCHW 1x1x256x256) -> MultiThreshold (-> UINT4) -> MaxPool(2x2,s2) -> global_out (UINT4, 1x1x128x128)."""
    in_shape = [1, CHANNELS, IFM_DIM, IFM_DIM]
    out_shape = [1, CHANNELS, IFM_DIM // 2, IFM_DIM // 2]
    n_thresh = 15  # UINT4 output: 16 levels -> 15 thresholds
    thresholds = np.sort(np.random.randint(1, 255, size=(CHANNELS, n_thresh)).astype(np.float32), axis=1)

    global_in = oh.make_tensor_value_info("global_in", TensorProto.FLOAT, in_shape)
    global_out = oh.make_tensor_value_info("global_out", TensorProto.FLOAT, out_shape)
    thresh_out = oh.make_tensor_value_info("thresh_out", TensorProto.FLOAT, in_shape)
    thresh_init = oh.make_tensor("thresholds", TensorProto.FLOAT, thresholds.shape, thresholds.flatten())

    thresh_node = oh.make_node(
        "MultiThreshold", ["global_in", "thresholds"], ["thresh_out"],
        domain="qonnx.custom_op.general", out_dtype="UINT4", out_scale=1.0, out_bias=0.0, data_layout="NCHW",
    )
    pool_node = oh.make_node(
        "MaxPool", ["thresh_out"], ["global_out"], kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )

    graph = oh.make_graph([thresh_node, pool_node], "thresh_pool_isolation", [global_in], [global_out],
                           value_info=[thresh_out], initializer=[thresh_init])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresh_pool_bd_isolation"))
    model.set_tensor_datatype("global_in", DataType["UINT8"])
    model.set_tensor_datatype("thresholds", DataType["UINT8"])
    model.set_tensor_datatype("thresh_out", DataType["UINT4"])
    model.set_tensor_datatype("global_out", DataType["UINT4"])
    return model


def main() -> None:
    m = build_model()
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(InferDataLayouts())
    m = m.transform(GiveUniqueNodeNames())
    m.save(os.path.join(out_dir, "step00_initial.onnx"))

    # same InferPool monkeypatch as finn_bottleneck_probe_build.py's pool_impl=="pool" route, and the
    # same canonical transform order as finn_enet_convert_to_hw_rtl_mvau.step_enet_convert_to_hw_rtl_mvau
    to_hw.InferStreamingMaxPool = getattr(to_hw, "InferPool", None) or getattr(to_hw, "InferPool_Batch")
    print("[isolation] pool lowered with", to_hw.InferStreamingMaxPool.__name__, flush=True)

    for trn in [
        to_hw.InferStreamingMaxPool,  # MaxPool (NCHW) -> Transpose + Im2Col(depthwise) + Pool + Transpose
        RoundAndClipThresholds,
        to_hw.InferThresholdingLayer,  # MultiThreshold -> standalone Thresholding
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,  # Im2Col -> ConvolutionInputGenerator
    ]:
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
