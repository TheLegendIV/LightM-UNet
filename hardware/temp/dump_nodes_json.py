"""Dump partition 1's node list as JSON (name + op_type, in graph order)
for use by analyze_vcd_stall.py."""
import sys
import os
import json

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

BUILD_TAG = "S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix"
ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
os.environ["FINN_BUILD_DIR"] = os.path.join(ENET_DIR, "finn_build_tmp", BUILD_TAG)

part_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out_path = sys.argv[2] if len(sys.argv) > 2 else ("/tmp/partition%d_nodes.json" % part_idx)

parent_ckpt = os.path.join(
    ENET_DIR,
    "finn_deployment_outputs",
    "S12_256_analytical_namefix_20261006_195156",
    "intermediate_models",
    "dataflow_parent_built.onnx",
)
parent = ModelWrapper(parent_ckpt)
sdp = [n for n in parent.get_nodes_by_op_type("StreamingDataflowPartition") if n.name == "GenericPartition_%d" % part_idx][0]
kernel_fn = getCustomOp(sdp).get_nodeattr("model")
model = ModelWrapper(kernel_fn)

nodes = []
for node in model.graph.node:
    entry = {"name": node.name, "op_type": node.op_type, "input": list(node.input), "output": list(node.output)}
    if node.op_type == "StreamingFIFO_rtl":
        inst = getCustomOp(node)
        entry["depth"] = inst.get_nodeattr("depth")
        entry["impl_style"] = inst.get_nodeattr("impl_style")
    nodes.append(entry)

with open(out_path, "w") as f:
    json.dump(nodes, f, indent=1)
print("wrote", len(nodes), "nodes to", out_path)
