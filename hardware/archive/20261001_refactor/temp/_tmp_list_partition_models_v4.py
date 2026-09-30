import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"
m = ModelWrapper(OUT + "/intermediate_models/dataflow_parent_built.onnx")
print("num nodes:", len(m.graph.node))
for i, n in enumerate(m.graph.node):
    print(i, "name=", n.name, "op_type=", n.op_type, "domain=", repr(n.domain))
