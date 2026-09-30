import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUT = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"

TARGETS = {
    6: ["StreamingFIFO_rtl_11", "StreamingFIFO_rtl_65", "StreamingFIFO_rtl_75", "StreamingFIFO_rtl_76",
        "StreamingFIFO_rtl_77", "StreamingFIFO_rtl_79", "StreamingFIFO_rtl_87", "StreamingFIFO_rtl_96", "StreamingFIFO_rtl_97"],
    7: ["StreamingFIFO_rtl_11", "StreamingFIFO_rtl_13", "StreamingFIFO_rtl_27", "StreamingFIFO_rtl_39"],
}

def fold_info(node):
    try:
        inst = getCustomOp(node)
        pe = inst.get_nodeattr("PE") if "PE" in [a.name for a in node.attribute] else None
        simd = inst.get_nodeattr("SIMD") if "SIMD" in [a.name for a in node.attribute] else None
        return f"PE={pe} SIMD={simd}"
    except Exception as e:
        return f"(no fold info: {e})"

for pidx, names in TARGETS.items():
    fn = OUT + f"/intermediate_models/supported_op_partitions/partition_{pidx}.onnx"
    m = ModelWrapper(fn)
    print(f"\n=== partition_{pidx} ===")
    for short in names:
        full = f"GenericPartition_{pidx}_{short}"
        node = None
        for n in m.graph.node:
            if n.name == full:
                node = n
                break
        if node is None:
            print(f"  {short}: NOT FOUND")
            continue
        in_tensor = node.input[0]
        out_tensor = node.output[0]
        producer = m.find_producer(in_tensor)
        consumer = m.find_consumer(out_tensor)
        depth = getCustomOp(node).get_nodeattr("depth")
        prod_str = f"{producer.op_type} '{producer.name}' [{fold_info(producer)}]" if producer else "GRAPH INPUT"
        cons_str = f"{consumer.op_type} '{consumer.name}' [{fold_info(consumer)}]" if consumer else "GRAPH OUTPUT"
        print(f"  {short} (depth={depth}):")
        print(f"     producer: {prod_str}")
        print(f"     consumer: {cons_str}")
