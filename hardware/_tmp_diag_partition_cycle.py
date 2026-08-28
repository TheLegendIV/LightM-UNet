"""One-off diagnostic: reproduce qonnx's PartitionFromLambda cycle check
node-by-node against the assign_stage_partition_ids_8way.onnx checkpoint,
to find exactly which partition/node/tensor triggers "cycle-free graph
violated: partition depends on itself" for the new Concat-based initial
block topology. Not part of the pipeline -- run manually, delete after use.
"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

MODEL_FILE = sys.argv[1]

model = ModelWrapper(MODEL_FILE)


def partition_of(node):
    try:
        return getCustomOp(node).get_nodeattr("partition_id")
    except Exception:
        return -1


all_nodes = list(model.graph.node)
partition_ids = sorted(set(partition_of(n) for n in all_nodes) - {-1})
print("partition ids found:", partition_ids)
print("total nodes:", len(all_nodes))

for pid in partition_ids:
    partition_nodes = [n for n in all_nodes if partition_of(n) == pid]
    p_in = []
    for node in partition_nodes:
        for in_tensor in node.input:
            has_initializer = in_tensor in [x.name for x in model.graph.initializer]
            has_producer = model.find_producer(in_tensor) is not None
            if not has_initializer and (has_producer or in_tensor not in [x.name for x in model.graph.input]):
                producer = model.find_producer(in_tensor)
                if producer is not None and partition_of(producer) != pid:
                    if in_tensor not in p_in:
                        p_in.append(in_tensor)

    to_check = [(t, model.find_producer(t)) for t in p_in]
    to_check = [(t, n) for (t, n) in to_check if n is not None]
    visited = set()
    found_cycle = False
    while to_check and not found_cycle:
        next_to_check = []
        for tensor_name, node in to_check:
            if node is None:
                continue
            key = node.name
            if key in visited:
                continue
            visited.add(key)
            if partition_of(node) == pid:
                print(f"\n*** CYCLE at partition {pid}: external input tensor "
                      f"'{tensor_name}' has ancestor node '{node.name}' "
                      f"(op_type={node.op_type}) which IS in partition {pid} ***")
                # print a short ancestry hint: this node's own inputs
                for i in node.input:
                    prod = model.find_producer(i)
                    print(f"    input '{i}' produced by "
                          f"{prod.name if prod else 'GRAPH_INPUT/INITIALIZER'} "
                          f"(partition={partition_of(prod) if prod else 'n/a'})")
                found_cycle = True
                break
            predecessors = model.find_direct_predecessors(node)
            if predecessors:
                for p in predecessors:
                    next_to_check.append((node.name, p))
        to_check = next_to_check
    if not found_cycle:
        print(f"partition {pid}: OK ({len(partition_nodes)} nodes, {len(p_in)} external inputs)")
