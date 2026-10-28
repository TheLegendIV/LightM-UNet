"""One-off: resolve partition 0's own standalone stitched-IP folded I/O
shapes + regenerate its pyverilator_vh header dir, so a standalone
backpressure-injecting rtlsim of JUST partition 0 can be built without
going through the full combine step."""
import sys
import os

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from finn.util.pyverilator import prepare_stitched_ip_for_verilator  # noqa: E402
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
sdp0 = [n for n in parent.get_nodes_by_op_type("StreamingDataflowPartition") if n.name == "GenericPartition_0"][0]
kernel_fn = getCustomOp(sdp0).get_nodeattr("model")
print("partition0 child model:", kernel_fn, flush=True)
kernel_model = ModelWrapper(kernel_fn)

vivado_stitch_proj_dir = prepare_stitched_ip_for_verilator(kernel_model)
print("stitch proj dir:", vivado_stitch_proj_dir, flush=True)

first_node = kernel_model.find_consumer(kernel_model.graph.input[0].name)
ishape_folded = getCustomOp(first_node).get_folded_input_shape()
last_node = kernel_model.find_producer(kernel_model.graph.output[0].name)
oshape_folded = getCustomOp(last_node).get_folded_output_shape()
print("ishape_folded", ishape_folded, "oshape_folded", oshape_folded, flush=True)

n_iters_per_input = int(np.prod(ishape_folded[:-1]))
n_iters_per_output = int(np.prod(oshape_folded[:-1]))
print("n_iters_per_input", n_iters_per_input, "n_iters_per_output", n_iters_per_output, flush=True)

with open("/tmp/partition0_bp_shapes.txt", "w") as f:
    f.write("%d %d\n" % (n_iters_per_input, n_iters_per_output))
    f.write(vivado_stitch_proj_dir + "\n")
print("wrote /tmp/partition0_bp_shapes.txt", flush=True)
