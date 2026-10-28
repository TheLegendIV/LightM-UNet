import json
from qonnx.core.modelwrapper import ModelWrapper

FN = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_analytical_v1_milpfold_8way_20261005_135935/intermediate_models/supported_op_partitions/partition_0_prefifo_autosize.onnx"

m = ModelWrapper(FN)

def real_producer(model, node):
    p = model.find_producer(node.input[0])
    while p is not None and p.op_type.startswith("StreamingFIFO"):
        p = model.find_producer(p.input[0])
    return p

def real_consumer(model, node):
    c = model.find_consumer(node.output[0])
    while c is not None and c.op_type.startswith("StreamingFIFO"):
        c = model.find_consumer(c.output[0])
    return c

print("ALL NODE OP_TYPES in partition 0, in order:")
for n in m.graph.node:
    print(f"  {n.name:35s} {n.op_type}")

print()
print("FIFO neighbor op_types:")
for n in m.graph.node:
    if not n.op_type.startswith("StreamingFIFO"):
        continue
    p = real_producer(m, n)
    c = real_consumer(m, n)
    p_s = f"{p.name}/{p.op_type}" if p is not None else "<GRAPH INPUT>"
    c_s = f"{c.name}/{c.op_type}" if c is not None else "<GRAPH OUTPUT>"
    depth = None
    for attr in n.attribute:
        if attr.name == "depth":
            depth = attr.i
    print(f"  {n.name:20s} depth={depth:>6}  producer=[{p_s}]  consumer=[{c_s}]")
