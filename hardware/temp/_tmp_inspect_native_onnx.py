import onnx

m = onnx.load("/home/thelegendiv/finn/notebooks/enet/quantEnet_original_int8.onnx")
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("node count:", len(m.graph.node))
print("op histogram:", dict(sorted(ops.items())))
print("\ninputs:")
for i in m.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(" ", i.name, dims)
print("outputs:")
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(" ", o.name, dims)
