from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import sys

path = sys.argv[1]
m = ModelWrapper(path)
for n in m.graph.node:
    if n.op_type in ("MVAU_rtl", "MVAU"):
        inst = getCustomOp(n)
        print(n.name, "noActivation=", inst.get_nodeattr("noActivation"))
