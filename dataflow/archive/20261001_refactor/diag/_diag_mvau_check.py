from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import sys

path = sys.argv[1]
name = sys.argv[2]
m = ModelWrapper(path)
n = next(x for x in m.graph.node if x.name == name)
inst = getCustomOp(n)
try:
    print(name, "noActivation=", inst.get_nodeattr("noActivation"))
except Exception as e:
    print(name, "no noActivation attr:", e)
print("  inputs=", list(n.input))
for inp in n.input:
    prod = m.find_producer(inp)
    if prod is not None:
        print("   <-", prod.op_type, prod.name)
for out in n.output:
    for c in m.find_consumers(out):
        print("   ->", c.op_type, c.name)
