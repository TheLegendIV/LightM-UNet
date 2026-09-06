import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

# Import the PATCHED module FIRST -- it sets up sys.path for finn/qonnx/brevitas itself
import finn_enet_build_decomposed_prelu as m

from qonnx.core.modelwrapper import ModelWrapper

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260810_012204/intermediate_models"

model = ModelWrapper(f"{BASE}/step_enet_streamline.onnx")
print("Re-running step_absorb_leftover_scale_before_matmul (PATCHED) from step_enet_streamline checkpoint...")
model = m.step_absorb_leftover_scale_before_matmul(model)

# Check MaxPool + Im2Col shapes now
n_bad = 0
for n in model.graph.node:
    if n.op_type == "MaxPoolNHWC":
        in_shape = model.get_tensor_shape(n.input[0])
        out_shape = model.get_tensor_shape(n.output[0])
        ok = (out_shape[1] == in_shape[1] // 2 and out_shape[2] == in_shape[2] // 2)
        if not ok:
            n_bad += 1
            print(f"  BAD MaxPoolNHWC {n.name}: in={in_shape} out={out_shape}")
    if n.op_type == "Im2Col":
        in_shape = model.get_tensor_shape(n.input[0])
        out_shape = model.get_tensor_shape(n.output[0])
        if out_shape[-1] == in_shape[-1]:  # no channel expansion at all = suspicious/stale
            n_bad += 1
            print(f"  SUSPICIOUS Im2Col {n.name}: in={in_shape} out={out_shape} (no channel expansion)")

print(f"\nTotal bad/suspicious nodes after patched step: {n_bad}")
