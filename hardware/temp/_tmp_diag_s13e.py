import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

path = sys.argv[1]
node_name = sys.argv[2]
model = ModelWrapper(path)
n = model.get_node_from_name(node_name)
print("node:", n.name, n.op_type, n.domain)
print("outputs:", list(n.output))
for o in n.output:
    for c in model.find_consumers(o):
        print(" consumer:", c.name, c.op_type, c.domain)
    print("  also check global output:", o in [x.name for x in model.graph.output])
