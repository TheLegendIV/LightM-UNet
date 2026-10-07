"""Check which partitions contain impl_style='vivado' StreamingFIFO_rtl nodes
(Xilinx FIFO_generator IP) vs. pure impl_style='rtl' FIFOs, to correlate with
the deadlock/pass pattern found via standalone rtlsim (1/5/6 deadlock,
0/2/3/4/7 pass). Vivado-style FIFOs may not have a real synthesizable
Verilog simulation model usable by Verilator -- prepare_for_stitched_ip_rtlsim
normally forces these to impl_style='rtl' + restitch before any rtlsim,
which our standalone test (reusing the stitched IP as originally built for
real hardware) did NOT do."""
import sys
import os

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

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

RESULT = {0: "PASS", 1: "DEADLOCK", 2: "PASS", 3: "PASS", 4: "PASS", 5: "DEADLOCK", 6: "DEADLOCK", 7: "PASS"}

for sdp in sdp_nodes:
    idx = int(sdp.name.split("_")[-1])
    kernel_fn = getCustomOp(sdp).get_nodeattr("model")
    model = ModelWrapper(kernel_fn)
    fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    vivado_fifos = []
    for fn in fifo_nodes:
        inst = getCustomOp(fn)
        if inst.get_nodeattr("impl_style") == "vivado":
            vivado_fifos.append((fn.name, inst.get_nodeattr("depth")))
    print(
        "Partition %d [%s]: %d total FIFOs, %d impl_style=vivado: %s"
        % (idx, RESULT.get(idx, "?"), len(fifo_nodes), len(vivado_fifos), vivado_fifos),
        flush=True,
    )
