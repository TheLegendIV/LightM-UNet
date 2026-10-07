"""Dump per-partition metadata (wrapper top-module name, stitched-IP dir,
clk period, folded I/O shapes) for ALL 8 partitions of the analytical u4
build, so a parallel per-partition rtlsim compile+run can be parameterized
correctly (confirms actual top-module name instead of guessing)."""
import sys
import os

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
import numpy as np  # noqa: E402

BUILD_TAG = "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix"
ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)

parent_ckpt = os.path.join(
    ENET_DIR,
    "finn_deployment_outputs",
    "S12_256_analytical_namefix_20261006_195156",
    "intermediate_models",
    "dataflow_parent_built.onnx",
)
parent = ModelWrapper(parent_ckpt)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
print("num partitions:", len(sdp_nodes), flush=True)

for sdp in sdp_nodes:
    name = sdp.name
    kernel_fn = getCustomOp(sdp).get_nodeattr("model")
    try:
        kernel_model = ModelWrapper(kernel_fn)
        wrapper_filename = kernel_model.get_metadata_prop("wrapper_filename")
        vivado_stitch_proj = kernel_model.get_metadata_prop("vivado_stitch_proj")
        clk_ns = kernel_model.get_metadata_prop("clk_ns")
        first_node = kernel_model.find_consumer(kernel_model.graph.input[0].name)
        last_node = kernel_model.find_producer(kernel_model.graph.output[0].name)
        ishape_folded = getCustomOp(first_node).get_folded_input_shape()
        oshape_folded = getCustomOp(last_node).get_folded_output_shape()
        n_iters_in = int(np.prod(ishape_folded[:-1]))
        n_iters_out = int(np.prod(oshape_folded[:-1]))
        has_srcs_txt = os.path.isfile(
            os.path.join(vivado_stitch_proj, "all_verilog_srcs.txt")
        ) if vivado_stitch_proj else False
        print(
            "%s | kernel=%s | wrapper_filename=%s | vivado_stitch_proj=%s | "
            "srcs_txt_exists=%s | clk_ns=%s | n_iters_in=%d | n_iters_out=%d"
            % (
                name,
                kernel_fn,
                wrapper_filename,
                vivado_stitch_proj,
                has_srcs_txt,
                clk_ns,
                n_iters_in,
                n_iters_out,
            ),
            flush=True,
        )
    except Exception as e:
        print("%s | ERROR: %r" % (name, e), flush=True)
