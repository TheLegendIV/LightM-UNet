import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

path = sys.argv[1]
model = ModelWrapper(path)

targets = ["MVAU_rtl_6", "MVAU_rtl_7", "MVAU_rtl_8", "MVAU_rtl_9", "MVAU_rtl_10",
           "ConvolutionInputGenerator_rtl_3"]
for name in targets:
    node = None
    for n in model.graph.node:
        if n.name == name:
            node = n
            break
    if node is None:
        print(f"{name}: NOT FOUND")
        continue
    inst = getCustomOp(node)
    print(f"=== {name} ({node.op_type}) ===")
    print(f"  input: {list(node.input)}")
    print(f"  output: {list(node.output)}")
    for attr in node.attribute:
        print(f"  attr {attr.name}: {inst.get_nodeattr(attr.name)}")
    # producer of this node's main input
    prod = model.find_producer(node.input[0])
    print(f"  producer of input[0]: {prod.name if prod else None} ({prod.op_type if prod else None})")
