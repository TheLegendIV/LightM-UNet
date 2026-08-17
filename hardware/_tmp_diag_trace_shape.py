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
    "step_enet_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_hw_ipgen",
]

for step in STEPS:
    path = f"{BASE}/{step}.onnx"
    try:
        model = ModelWrapper(path)
    except Exception as e:
        print(f"=== {step}: FAILED TO LOAD ({e}) ===")
        continue
    graph = model.graph
    # find the maxpool node (name contains "StreamingMaxPool" and ends in _0, or plain "MaxPool")
    maxpool_node = None
    for n in graph.node:
        if "MaxPool" in n.op_type and (n.name.endswith("_0") or n.name == "StreamingMaxPool_0" or n.name == "MaxPool_0"):
            maxpool_node = n
            break
    if maxpool_node is None:
        # dataflow partition step wraps everything in a StreamingDataflowPartition; look inside
        print(f"=== {step}: no top-level maxpool-ish node found (checked {len(graph.node)} nodes, op_types sample: {sorted(set(n.op_type for n in graph.node))[:10]}) ===")
        continue
    out_tensor = maxpool_node.output[0]
    consumer = model.find_consumer(out_tensor)
    print(f"=== {step} ===")
    print(f"  maxpool node: {maxpool_node.name} ({maxpool_node.op_type})")
    try:
        print(f"    out normal shape: {model.get_tensor_shape(out_tensor)}")
    except Exception as e:
        print(f"    out normal shape: ERROR {e}")
    if consumer is not None:
        print(f"  consumer: {consumer.name} ({consumer.op_type})")
        try:
            print(f"    consumer in0 shape (get_tensor_shape): {model.get_tensor_shape(consumer.input[0])}")
        except Exception as e:
            print(f"    consumer in0 shape: ERROR {e}")
        try:
            inst = getCustomOp(consumer)
            if "MVAU" in consumer.op_type or "MatrixVectorActivation" in consumer.op_type:
                print(f"    consumer ifm_dim/ofm_dim/ich/och nodeattrs: {inst.get_nodeattr('MW') if inst.get_nodeattr_types().get('MW') else ''}")
                for k in ("numInputVectors", "IFMDim", "OFMDim", "MH", "MW"):
                    try:
                        print(f"      {k} = {inst.get_nodeattr(k)}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"    custom op inspect error: {e}")
    else:
        print("  no consumer found (fork or output?)")
