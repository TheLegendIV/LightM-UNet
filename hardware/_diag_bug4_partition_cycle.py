"""Diagnostic (not part of the build pipeline): replicate
CreateDataflowPartition's assign_partition_id + PartitionFromLambda's
cycle-free check on assign_stage_partition_ids_8way.onnx, to find exactly
which partition_id and node triggers "cycle-free graph violated: partition
depends on itself"."""
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8_largefifo_rtlsim_20260909_002048/intermediate_models/"
model = ModelWrapper(BASE + "assign_stage_partition_ids_8way.onnx")


def assign_partition_id(node):
    if node.op_type in ("GenericPartition", "StreamingDataflowPartition"):
        return -1
    backend = get_by_name(node.attribute, "backend")
    if backend is not None and backend.s.decode("UTF-8") == "fpgadataflow":
        assigned_partition = get_by_name(node.attribute, "partition_id")
        if assigned_partition is not None:
            return assigned_partition.i
        return 0
    return -1


all_nodes = list(model.graph.node)
partition_ids = sorted(set(assign_partition_id(n) for n in all_nodes) - {-1})
print("partition ids:", partition_ids)
print("nodes with pid=-1 (non-fpgadataflow, excluded):", len([n for n in all_nodes if assign_partition_id(n) == -1]))
for n in all_nodes:
    if assign_partition_id(n) == -1:
        print("  excluded:", n.name, n.op_type)

for partition_id in partition_ids:
    partition_nodes = [n for n in all_nodes if assign_partition_id(n) == partition_id]
    non_partition_nodes = [n for n in all_nodes if n not in partition_nodes]
    p_node_names = {n.name for n in partition_nodes}

    p_in = []
    for node in partition_nodes:
        for in_tensor in node.input:
            has_initializer = in_tensor in [x.name for x in model.graph.initializer]
            producer = model.find_producer(in_tensor)
            has_producer_in_partition = producer is not None and producer.name in p_node_names
            if not has_initializer and not has_producer_in_partition:
                if in_tensor not in p_in:
                    p_in.append(in_tensor)

    to_check = [model.find_producer(x) for x in p_in]
    seen = set()
    while len(to_check) > 0:
        next_to_check = []
        for node in to_check:
            if node is None or node.name in seen:
                continue
            seen.add(node.name)
            pid = assign_partition_id(node)
            if pid == partition_id:
                print(f"CYCLE FOUND: partition_id={partition_id} node={node.name} op_type={node.op_type}")
            predecessors = model.find_direct_predecessors(node)
            if predecessors is not None:
                next_to_check.extend(predecessors)
        to_check = next_to_check
print("done")
