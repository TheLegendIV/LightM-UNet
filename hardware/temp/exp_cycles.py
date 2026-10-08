"""Print the slowest nodes (get_exp_cycles) of a stitched partition ONNX.

Usage: python3 exp_cycles.py <stitched.onnx> [top_n]
"""
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = []
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        continue
    try:
        rows.append((getCustomOp(n).get_exp_cycles(), n.name))
    except Exception:
        pass
rows.sort(reverse=True)
for c, name in rows[:top]:
    print("%-9d %s" % (c, name))
