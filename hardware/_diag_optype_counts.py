from qonnx.core.modelwrapper import ModelWrapper
import sys
from collections import Counter

path = sys.argv[1]
m = ModelWrapper(path)
g = m.graph
c = Counter(n.op_type for n in g.node)
for op, cnt in sorted(c.items()):
    print(f"{cnt:4d}  {op}")
