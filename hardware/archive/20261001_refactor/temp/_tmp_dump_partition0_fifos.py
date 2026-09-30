import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

model = ModelWrapper(sys.argv[1])
total_blocks = 0
for node in model.graph.node:
    if "StreamingFIFO" not in node.op_type:
        continue
    inst = getCustomOp(node)
    depth = inst.get_nodeattr("depth")
    impl = inst.get_nodeattr("impl_style")
    ram = inst.get_nodeattr("ram_style") if "ram_style" in [a.name for a in node.attribute] else "n/a"
    width = inst.get_folded_output_shape()
    print(f"{node.name:40s} depth={depth:6d} impl={impl:6s} ram_style={ram}")
print(f"\ntotal FIFO nodes: {sum(1 for n in model.graph.node if 'StreamingFIFO' in n.op_type)}")
