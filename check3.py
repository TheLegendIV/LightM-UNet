from qonnx.core.modelwrapper import ModelWrapper
from collections import Counter

path = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_091715/intermediate_models/step_enet_convert_to_hw.onnx"
m = ModelWrapper(path)
weight_nodes = [n for n in m.graph.node if n.op_type in ("MatrixVectorActivation","MVAU","VVAU")]
print("total weight nodes:", len(weight_nodes))

# for each node, find its weight input tensor name (2nd input typically)
tensor_names = []
for n in weight_nodes:
    # find inputs that are initializers
    for inp in n.input:
        if m.get_initializer(inp) is not None:
            tensor_names.append((n.name, inp))
            break

names_only = [t[1] for t in tensor_names]
dupes = {k: v for k, v in Counter(names_only).items() if v > 1}
print("duplicate weight tensor names (shared across multiple nodes):", dupes)
print("total unique weight tensors used:", len(set(names_only)), "vs total nodes:", len(tensor_names))