"""List all node types/names in partition 1's child model, flagging any
fork/duplicate-stream node (user's hypothesis: a 'fork fifo' is the
bottleneck) and its neighboring FIFOs."""
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
sdp1 = [n for n in parent.get_nodes_by_op_type("StreamingDataflowPartition") if n.name == "GenericPartition_1"][0]
kernel_fn = getCustomOp(sdp1).get_nodeattr("model")
model = ModelWrapper(kernel_fn)

print("=== partition 1 node list (in graph order) ===", flush=True)
for i, node in enumerate(model.graph.node):
    ins = list(node.input)
    outs = list(node.output)
    extra = ""
    if "Duplicate" in node.op_type or "fork" in node.name.lower() or "Fork" in node.op_type:
        extra = "  <<<< FORK/DUPLICATE CANDIDATE"
    if node.op_type == "StreamingFIFO_rtl":
        inst = getCustomOp(node)
        depth = inst.get_nodeattr("depth")
        impl_style = inst.get_nodeattr("impl_style")
        extra += "  [depth=%s impl_style=%s]" % (depth, impl_style)
    print("%3d  %-28s name=%-40s in=%s out=%s%s" % (i, node.op_type, node.name, ins, outs, extra), flush=True)

print(flush=True)
print("=== fan-out check: any tensor consumed by >1 node? ===", flush=True)
from collections import defaultdict  # noqa: E402
consumers = defaultdict(list)
for node in model.graph.node:
    for inp in node.input:
        consumers[inp].append(node.name)
for tensor, cons in consumers.items():
    if len(cons) > 1:
        print("tensor %s consumed by: %s" % (tensor, cons), flush=True)
