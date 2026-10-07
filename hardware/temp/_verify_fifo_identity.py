import json
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

def build_edge_maps(model):
    producer_of = {}
    consumers_of = {}
    for node in model.graph.node:
        for out in node.output:
            producer_of[out] = node
        for inp in node.input:
            consumers_of.setdefault(inp, []).append(node)
    return producer_of, consumers_of

def fifo_info(model, name):
    producer_of, consumers_of = build_edge_maps(model)
    for node in model.graph.node:
        if node.name == name:
            inst = getCustomOp(node)
            depth = inst.get_nodeattr("depth")
            prod = producer_of.get(node.input[0])
            cons = consumers_of.get(node.output[0], [])
            return {
                "depth": depth,
                "producer": f"{prod.op_type}:{prod.name}" if prod else None,
                "consumers": [f"{c.op_type}:{c.name}" for c in cons],
            }
    return None

a = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1/partition6_refix_stitched.onnx')
b = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_autosize_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_autosize1/partition6_autosize_postfifo.onnx')

for name in ["StreamingFIFO_rtl_4", "StreamingFIFO_rtl_5", "StreamingFIFO_rtl_6", "StreamingFIFO_rtl_9", "StreamingFIFO_rtl_49"]:
    print(f"--- {name} ---")
    print("  analytical:", fifo_info(a, name))
    print("  autosize:  ", fifo_info(b, name))
