import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental/src")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

BASE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260810_012204/intermediate_models"

STEPS = [
    "step_qonnx_to_finn",
    "_fixup_degenerate_signed_bias",
    "step_fuse_leaky_relu_to_threshold",
    "step_enet_streamline",
    "step_absorb_leftover_scale_before_matmul",
    "step_fuse_forked_dequant_into_duplicate_threshold",
    "step_enet_convert_to_hw",
]

for step in STEPS:
    path = f"{BASE}/{step}.onnx"
    try:
        model = ModelWrapper(path)
    except Exception as e:
        print(f"=== {step}: FAILED TO LOAD ({e}) ===")
        continue
    graph = model.graph
    maxpool_node = None
    for n in graph.node:
        if "MaxPool" in n.op_type:
            maxpool_node = n
            break
    print(f"=== {step} ({len(graph.node)} nodes) ===")
    if maxpool_node is None:
        print("  NO maxpool node found")
        continue
    print(f"  maxpool node: {maxpool_node.name} ({maxpool_node.op_type})")
    print(f"    attrs: {[(a.name, a.i if a.type==2 else (list(a.ints) if a.type==7 else a.s)) for a in maxpool_node.attribute]}")
    try:
        print(f"    in shape:  {model.get_tensor_shape(maxpool_node.input[0])}")
    except Exception as e:
        print(f"    in shape: ERROR {e}")
    try:
        print(f"    out shape: {model.get_tensor_shape(maxpool_node.output[0])}")
    except Exception as e:
        print(f"    out shape: ERROR {e}")
