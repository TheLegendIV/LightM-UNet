import sys
from qonnx.core.modelwrapper import ModelWrapper

f = sys.argv[1]
m = ModelWrapper(f)
ops = sorted(set(n.op_type for n in m.graph.node))
print(ops)
