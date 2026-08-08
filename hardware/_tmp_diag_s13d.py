"""Reproduce qonnx's PartitionFromLambda backward-walk logic manually
(without asserting) to find every non-HW ancestor node reachable from a
partition-external input tensor that ITSELF is (transitively) fed by an
HW node -- i.e. the actual node(s) that would trip
"cycle-free graph violated: partition depends on itself"."""
import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

path = sys.argv[1]
model = ModelWrapper(path)
graph = model.graph


def is_hw(node):
    return node.domain.startswith("finn.custom_op") and "fpgadataflow" in node.domain


def partition_id(node):
    return 0 if is_hw(node) else -1


# Partition-external inputs: tensors consumed by an HW node but produced by
# a non-HW node (or produced nowhere -- true graph input).
p_ins = []
for node in graph.node:
    if not is_hw(node):
        continue
    for inp in node.input:
        producer = model.find_producer(inp)
        if producer is not None and not is_hw(producer):
            p_ins.append((inp, node.name, producer.name))

print(f"Found {len(p_ins)} partition-external input tensors (HW node fed by non-HW producer)")
for t, consumer, producer in p_ins:
    print(f"  tensor={t}  consumer(HW)={consumer}  producer(non-HW)={producer}")

# Now walk backward from each such tensor through non-HW predecessors,
# looking for any ancestor that IS HW (partition_id == 0) -- that's the
# self-dependency trigger.
print()
for t, consumer, producer in p_ins:
    visited = set()
    stack = [t]
    found_hw_ancestor = None
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        prod = model.find_producer(cur)
        if prod is None:
            continue
        if is_hw(prod):
            found_hw_ancestor = prod.name
            break
        for inp in prod.input:
            stack.append(inp)
    if found_hw_ancestor:
        print(f"CYCLE: tensor={t} (consumer={consumer}) backward-reaches HW node "
              f"{found_hw_ancestor} via non-HW ancestor chain -- "
              f"visited {len(visited)} tensors")
    else:
        print(f"OK: tensor={t} (consumer={consumer}) -- no HW ancestor found "
              f"in its {len(visited)}-tensor non-HW backward closure")
