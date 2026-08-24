import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

fn = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224/intermediate_models/step_build_all_partitions_capped.onnx"
model = ModelWrapper(fn)

print("=== all nodes (op_type, name, inputs, outputs) ===")
for n in model.graph.node:
    print(n.op_type, n.name, "IN:", list(n.input), "OUT:", list(n.output))

print()
print("=== per-SDP-node input/output producer/consumer map ===")
sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
for node in sdp_nodes:
    print("---", node.name, "---")
    for i, iname in enumerate(node.input):
        prod = model.find_producer(iname)
        print("  in[%d] %s <- produced by %s" % (i, iname, prod.name if prod else "GRAPH INPUT"))
    for i, oname in enumerate(node.output):
        cons = model.find_consumers(oname)
        cons_names = [c.name for c in cons] if cons else []
        print("  out[%d] %s -> consumed by %s" % (i, oname, cons_names if cons_names else "GRAPH OUTPUT"))
