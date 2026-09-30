import sys
from qonnx.core.modelwrapper import ModelWrapper

onnx_path = sys.argv[1]
targets = sys.argv[2].split(",")
model = ModelWrapper(onnx_path)

for t in targets:
    node = None
    for n in model.graph.node:
        if n.name == t:
            node = n
            break
    if node is None:
        print(t, "NOT FOUND")
        continue
    prod = model.find_producer(node.input[0])
    cons = model.find_consumers(node.output[0])
    print(f"{t}: input from -> {prod.name if prod else None} ({prod.op_type if prod else None})"
          f" | output to -> {[(c.name, c.op_type) for c in (cons or [])]}")
