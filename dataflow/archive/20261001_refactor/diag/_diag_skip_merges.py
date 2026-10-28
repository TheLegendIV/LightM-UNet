from qonnx.core.modelwrapper import ModelWrapper
import sys

path = sys.argv[1]
m = ModelWrapper(path)
g = m.graph
merge_ops = {"Add", "Concat"}
for n in g.node:
    if n.op_type in merge_ops:
        print("MERGE", n.op_type, n.name, "inputs=", list(n.input))
        for inp in n.input:
            prod = m.find_producer(inp)
            if prod is not None:
                print("   producer of", inp, "->", prod.op_type, prod.name)
            else:
                print("   producer of", inp, "-> (graph input / none)")
