import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental/src")

from qonnx.core.modelwrapper import ModelWrapper

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260810_012204/intermediate_models"

before = ModelWrapper(f"{BASE}/step_enet_streamline.onnx")
after = ModelWrapper(f"{BASE}/step_absorb_leftover_scale_before_matmul.onnx")

before_im2col = {n.name: n for n in before.graph.node if n.op_type == "Im2Col"}
after_im2col = {n.name: n for n in after.graph.node if n.op_type == "Im2Col"}

n_diff = 0
for name, n_after in after_im2col.items():
    n_before = before_im2col.get(name)
    if n_before is None:
        print(f"{name}: NEW node (not present before)")
        continue
    try:
        shp_before = before.get_tensor_shape(n_before.output[0])
    except Exception as e:
        shp_before = f"ERR {e}"
    try:
        shp_after = after.get_tensor_shape(n_after.output[0])
    except Exception as e:
        shp_after = f"ERR {e}"
    if shp_before != shp_after:
        n_diff += 1
        print(f"{name}: out shape changed {shp_before} -> {shp_after}  (in: before={before.get_tensor_shape(n_before.input[0])} after={after.get_tensor_shape(n_after.input[0])})")

print(f"\nTotal Im2Col nodes: before={len(before_im2col)} after={len(after_im2col)}, changed={n_diff}")
