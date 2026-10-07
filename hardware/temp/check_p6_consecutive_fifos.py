"""Diagnostic: list every StreamingFIFO_rtl node in partition 6's stitched graph and flag
any producer->consumer pair that are BOTH FIFOs (back-to-back, no compute node between).
Read-only, no rebuild.

Run:
    docker exec -e HOME=/tmp/home_dir finn_persistent python3 \\
        /home/thelegendiv/finn/notebooks/enet/check_p6_consecutive_fifos.py
"""
import json
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

ENET_DIR = "/home/thelegendiv/finn/notebooks/enet"
STITCHED = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    "partition6_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1",
    "partition6_refix_stitched.onnx",
)
FIFO_REPORT = os.path.join(
    ENET_DIR, "finn_deployment_outputs",
    "partition6_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1",
    "fifo_force_report_partition_6.json",
)

m = ModelWrapper(STITCHED)
graph = m.graph

node_by_output = {}
for n in graph.node:
    for o in n.output:
        node_by_output[o] = n

fifo_nodes = [n for n in graph.node if n.op_type == "StreamingFIFO_rtl"]
print(f"total nodes={len(graph.node)}  fifo nodes={len(fifo_nodes)}", flush=True)

# build a name->depth map from the fifo force report (role-forced ones)
forced_by_consumer = {}
if os.path.isfile(FIFO_REPORT):
    with open(FIFO_REPORT) as f:
        report = json.load(f)
    for e in report:
        forced_by_consumer[(e.get("producer_node"), e.get("consumer_node"))] = e

consecutive_pairs = []
for n in graph.node:
    if n.op_type != "StreamingFIFO_rtl":
        continue
    for inp in n.input:
        prod = node_by_output.get(inp)
        if prod is not None and prod.op_type == "StreamingFIFO_rtl":
            consecutive_pairs.append((prod, n))

print(f"consecutive FIFO->FIFO pairs found: {len(consecutive_pairs)}", flush=True)
for prod, cons in consecutive_pairs:
    prod_depth = getCustomOp(prod).get_nodeattr("depth")
    cons_depth = getCustomOp(cons).get_nodeattr("depth")
    # what feeds the first fifo, and what consumes the second
    upstream_of_prod = [node_by_output.get(i) for i in prod.input]
    downstream_of_cons = [c for c in graph.node for ci in c.input if ci in cons.output]
    up_name = upstream_of_prod[0].name if upstream_of_prod and upstream_of_prod[0] is not None else "GRAPH_INPUT"
    up_op = upstream_of_prod[0].op_type if upstream_of_prod and upstream_of_prod[0] is not None else "?"
    down_names = sorted(set(c.name for c in downstream_of_cons))
    print(
        f"  {up_op}:{up_name} -> FIFO:{prod.name}(depth={prod_depth}) -> "
        f"FIFO:{cons.name}(depth={cons_depth}) -> {down_names}",
        flush=True,
    )

# Also print full node sequence (op_type + name) for manual topology inspection
print("\n--- full node op_type list (topological order as stored) ---", flush=True)
for i, n in enumerate(graph.node):
    print(f"{i:3d} {n.op_type:30s} {n.name}", flush=True)
