import onnx
import sys

path = sys.argv[1]
m = onnx.load(path)
for i, n in enumerate(m.graph.node):
    attrs = ""
    print(i, n.op_type, n.name)
