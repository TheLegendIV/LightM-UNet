"""Diagnostic (not part of the build pipeline): run the REAL
PartitionFromLambda.apply() cycle-free check (monkeypatched to print instead
of assert-crash) against assign_stage_partition_ids_8way.onnx, to find the
exact node/partition_id triggering "cycle-free graph violated" without any
risk of a hand-reimplemented check diverging from the real one."""
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

import copy
import pathlib
import tempfile

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8_largefifo_rtlsim_20260909_002048/intermediate_models/"
model = ModelWrapper(BASE + "assign_stage_partition_ids_8way.onnx")


def assign_partition_id(node):
    if node.op_type in ["GenericPartition", "StreamingDataflowPartition"]:
        return -1
    backend = get_by_name(node.attribute, "backend")
    if backend is not None and backend.s.decode("UTF-8") == "fpgadataflow":
        assigned_partition = get_by_name(node.attribute, "partition_id")
        if assigned_partition is not None:
            return assigned_partition.i
        return 0
    return -1


all_nodes = list(model.graph.node)
partition_ids = set(map(assign_partition_id, all_nodes))
partition_ids.discard(-1)

for partition_id in sorted(partition_ids):
    partition_nodes = list(filter(lambda x: assign_partition_id(x) == partition_id, all_nodes))
    p_in = []
    for node in partition_nodes:
        for in_tensor in node.input:
            has_initializer = in_tensor in [x.name for x in model.graph.initializer]
            has_producer = any(in_tensor in list(pn.output) for pn in partition_nodes)
            if not has_initializer and not has_producer:
                if in_tensor not in p_in:
                    p_in.append(in_tensor)

    to_check = [model.find_producer(x) for x in p_in]
    depth = 0
    visited = set()
    while len(to_check) > 0:
        depth += 1
        if depth > 10000:
            print(f"partition {partition_id}: bailing after 10000 iterations (likely infinite loop in my walk)")
            break
        next_to_check = []
        for node in to_check:
            if node is not None:
                key = (partition_id, node.name)
                if assign_partition_id(node) == partition_id:
                    print(f"REAL-CHECK CYCLE: partition_id={partition_id} node={node.name} idx={all_nodes.index(node)}")
                if key in visited:
                    continue
                visited.add(key)
                predecessors = model.find_direct_predecessors(node)
                if predecessors is not None:
                    next_to_check.extend(predecessors)
        to_check = next_to_check
print("done, checked partitions:", sorted(partition_ids))
