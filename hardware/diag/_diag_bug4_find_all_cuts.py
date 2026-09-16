"""Diagnostic (not part of the build pipeline): iteratively discover ALL
partition-boundary cuts needed to make assign_stage_partition_ids_8way's
5-marker-based (down1/down2/q2/q3/q4/up4/up5) scheme fully cycle-free for
this specific model (which has a Concat-based initial block whose own
internal fork doesn't close via a literal AddStreams, plus down1/down2's
own MaxPool-shortcut forks) -- feedback-driven: run the REAL
PartitionFromLambda cycle check, and for every violation found, insert an
extra cut right after the offending (predecessor-side) node, then repeat
until the model partitions cleanly. Prints the final cut list to hard-code
into finn_stage_partition.py's boundary computation."""
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from qonnx.transformation.general import SortGraph
from qonnx.custom_op.registry import getCustomOp
import finn_stage_partition as fsp

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_partitioned8way_quantEnet_12_dense_relu_warmstart150ep_alpha025_trained_int8_largefifo_rtlsim_20260909_002048/intermediate_models/"
model = ModelWrapper(BASE + "step_enet_convert_to_hw.onnx")
model = model.transform(SortGraph())
nodes = list(model.graph.node)

down1_start, down2_start, up4_start, up5_start = fsp.find_stage_boundaries(model)
q2_start, q3_start, q4_start = fsp.find_stage23_quarter_boundaries(down2_start, up4_start)
cuts = sorted(set([0, down1_start, down2_start, q2_start, q3_start, q4_start, up4_start, up5_start, len(nodes)]))


def pid_of(idx):
    for i in range(len(cuts) - 1):
        if cuts[i] <= idx < cuts[i + 1]:
            return i
    return len(cuts) - 2


def assign_partition_id(node):
    if node.op_type in ("GenericPartition", "StreamingDataflowPartition"):
        return -1
    backend = get_by_name(node.attribute, "backend")
    if backend is not None and backend.s.decode("UTF-8") == "fpgadataflow":
        p = get_by_name(node.attribute, "partition_id")
        return p.i if p is not None else 0
    return -1


for round_num in range(30):
    for idx, node in enumerate(nodes):
        if not fsp._is_fpgadataflow_node(node):
            continue
        getCustomOp(node).set_nodeattr("partition_id", pid_of(idx))

    partition_ids = sorted(set(map(assign_partition_id, nodes)) - {-1})
    violation = None
    for partition_id in partition_ids:
        partition_nodes = [n for n in nodes if assign_partition_id(n) == partition_id]
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
        visited = set()
        found = None
        while len(to_check) > 0 and found is None:
            next_to_check = []
            for node in to_check:
                if node is None or node.name in visited:
                    continue
                visited.add(node.name)
                if assign_partition_id(node) == partition_id:
                    found = node
                    break
                predecessors = model.find_direct_predecessors(node)
                if predecessors is not None:
                    next_to_check.extend(predecessors)
            to_check = next_to_check
        if found is not None:
            violation = (partition_id, found)
            break
    if violation is None:
        print(f"CONVERGED after {round_num} extra cuts")
        print("final cuts:", cuts)
        break
    partition_id, bad_node = violation
    bad_idx = fsp._node_index(model, bad_node)
    newcut = bad_idx + 1
    print(f"round {round_num}: violation in partition {partition_id} at node {bad_node.name} (idx {bad_idx}) -> inserting cut at {newcut}")
    if newcut in cuts:
        print("ERROR: cut already exists, cannot converge this way")
        break
    cuts.append(newcut)
    cuts.sort()
else:
    print("DID NOT CONVERGE within 30 rounds")
