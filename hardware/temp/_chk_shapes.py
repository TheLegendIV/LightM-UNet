import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
m = ModelWrapper(sys.argv[1]).transform(InferShapes())
print("ok:", m.check_all_tensor_shapes_specified(fix_missing_init_shape=True))
print("ok2:", m.check_all_tensor_shapes_specified())
for n in m.graph.node:
    for t in list(n.input) + list(n.output):
        if m.get_tensor_shape(t) is None:
            print("NONE", n.op_type, n.name, repr(t))
