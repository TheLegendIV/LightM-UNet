import os
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import finn_enet_build_decomposed_prelu as m  # patched module; sets up sys.path

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from qonnx.core.modelwrapper import ModelWrapper

MODEL_NAME = "quantEnet_s19_double_mid_int8"
MODEL_FILE = os.path.join(m.ENET_DIR, f"{MODEL_NAME}.onnx")
OUTPUT_DIR = "/tmp/verify_fix_fast_build"

steps = [
    "step_qonnx_to_finn",
    m.step_enet_tidy,
    m.step_fuse_leaky_relu_to_threshold,
    m.step_enet_streamline,
    m.step_absorb_leftover_scale_before_matmul,
    m.step_fuse_forked_dequant_into_duplicate_threshold,
    m._fixup_degenerate_signed_bias,
    m.step_enet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

cfg = DataflowBuildConfig(
    output_dir          = OUTPUT_DIR,
    mvau_wwidth_max     = 80,
    target_fps          = 250,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xczu7ev-ffvc1156-2-e",
    auto_fifo_strategy  = build_cfg.AutoFIFOSizingMethod("largefifo_rtlsim"),
    steps               = steps,
    generate_outputs    = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    save_intermediate_models = True,
)

print("Running fast (no-HLS) pipeline up through step_generate_estimate_reports with the PATCHED shape fix...")
build.build_dataflow_cfg(MODEL_FILE, cfg)

print("\nLoading resulting dataflow-partition model and running InsertDWC+InsertFIFO check...")
parent = ModelWrapper(os.path.join(OUTPUT_DIR, "intermediate_models", "step_minimize_bit_width.onnx"))
# The dataflow partition is a separate child model; find it
sdp_node = None
for n in parent.graph.node:
    if n.op_type == "StreamingDataflowPartition":
        sdp_node = n
        break
if sdp_node is not None:
    from qonnx.custom_op.registry import getCustomOp
    inst = getCustomOp(sdp_node)
    child_path = inst.get_nodeattr("model")
    print(f"  child model: {child_path}")
    child = ModelWrapper(child_path)
else:
    child = parent

try:
    child = child.transform(InsertDWC())
    child = child.transform(InsertFIFO(create_shallow_fifos=True))
    print("\nSUCCESS: InsertDWC + InsertFIFO completed with NO assertion error!")
except AssertionError as e:
    print(f"\nSTILL FAILING: {e}")
