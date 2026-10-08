"""Compare two partition-tagged preamble checkpoints node by node (op_type, name, partition id, attrs)."""
import sys
from qonnx.core.modelwrapper import ModelWrapper

a, b = (ModelWrapper(p) for p in sys.argv[1:3])
na, nb = list(a.graph.node), list(b.graph.node)
print("nodes:", len(na), len(nb))


def key(n):
    attrs = tuple(sorted((x.name, str(x.i) if x.type == 2 else str(x.s) if x.type == 3 else "") for x in n.attribute))
    return (n.op_type, n.name, attrs)


diff = [(i, key(x)[:2], key(y)[:2]) for i, (x, y) in enumerate(zip(na, nb)) if key(x) != key(y)]
print("differing nodes:", len(diff))
for d in diff[:10]:
    print(d)
print("initializers equal:", {i.name: i.raw_data for i in a.graph.initializer} == {i.name: i.raw_data for i in b.graph.initializer})
