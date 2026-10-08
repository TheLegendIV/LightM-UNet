import sys, numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.datatype import DataType

def load(p):
    m = ModelWrapper(p).transform(InferShapes())
    for node in m.graph.node:
        for i, t in enumerate(node.input):
            if t == "":
                node.input[i] = f"{node.name}_empty{i}"
                m.set_initializer(node.input[i], np.zeros((0,), dtype=np.float32))
    m.check_all_tensor_shapes_specified(fix_missing_init_shape=True)
    m.set_tensor_datatype(m.graph.output[0].name, DataType['FLOAT32'])
    return m

ref = np.load(sys.argv[1]); a, b = load(sys.argv[2]), load(sys.argv[3])
x = ref["u"][0].astype(np.float32).reshape(1, 1, 256, 256)
ca = execute_onnx(a, {a.graph.input[0].name: x}, return_full_exec_context=True)
cb = execute_onnx(b, {b.graph.input[0].name: x}, return_full_exec_context=True)
bad = 0
for n in b.graph.node:
    for o in n.output:
        if o in ca and np.shape(ca[o]) == np.shape(cb[o]) and not np.allclose(ca[o], cb[o], atol=1e-4):
            print("FIRST MISMATCH", n.op_type, n.name, o, "maxabs", float(np.abs(ca[o] - cb[o]).max()))
            bad += 1
            break
    if bad:
        break
print("common tensors", len([o for o in cb if o in ca]), "mismatch found" if bad else "no mismatch among common")
print("out raw", ca[a.graph.output[0].name].shape, "out b", cb[b.graph.output[0].name].shape)

