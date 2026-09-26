from qonnx.core.modelwrapper import ModelWrapper
import sys

path = sys.argv[1]
m = ModelWrapper(path)
for n in m.graph.node:
    if n.op_type not in ("MVAU_rtl", "MVAU"):
        continue
    for out in n.output:
        consumers = m.find_consumers(out)
        cons_desc = [f"{c.op_type}:{c.name}" for c in consumers] if consumers else ["(graph output)"]
        print(f"{n.name:30s} n_consumers={len(consumers)} -> {cons_desc}")
