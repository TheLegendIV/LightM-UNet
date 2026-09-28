import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"
m = ModelWrapper(OUT + "/intermediate_models/dataflow_parent_built.onnx")
for n in m.graph.node:
    if n.op_type in ("Transpose", "Mul"):
        print(n.op_type, n.name, "inputs=", list(n.input), "outputs=", list(n.output))
        for a in n.attribute:
            print("   attr", a.name, a)
print("graph.input:", [x.name for x in m.graph.input])
print("graph.output:", [x.name for x in m.graph.output])
