import onnx
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "quantEnet_s19_double_mid_int8.onnx"
m = onnx.load(path)
for i, n in enumerate(m.graph.node):
    print(i, n.op_type, n.name)
