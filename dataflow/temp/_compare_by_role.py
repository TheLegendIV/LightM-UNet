from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

def build_maps(model):
    producer_of = {}
    consumers_of = {}
    for node in model.graph.node:
        for out in node.output:
            producer_of[out] = node
        for inp in node.input:
            consumers_of.setdefault(inp, []).append(node)
    return producer_of, consumers_of

def fifo_depth_between(model, producer_name, consumer_name):
    """Find FIFO depth(s) on any path directly connecting producer_name's output
    to consumer_name's input (possibly through a chain of FIFOs)."""
    producer_of, consumers_of = build_maps(model)
    name_to_node = {n.name: n for n in model.graph.node}
    prod = name_to_node[producer_name]
    cons = name_to_node[consumer_name]
    # walk forward from producer until we reach consumer, collecting FIFO depths
    frontier = [prod]
    visited = set()
    chain = []
    for _ in range(10):
        new_frontier = []
        for node in frontier:
            for out_t in node.output:
                for nxt in consumers_of.get(out_t, []):
                    if nxt.name == cons.name:
                        return chain
                    if "StreamingFIFO" in nxt.op_type and nxt.name not in visited:
                        visited.add(nxt.name)
                        inst = getCustomOp(nxt)
                        chain.append((nxt.name, inst.get_nodeattr("depth")))
                        new_frontier.append(nxt)
        frontier = new_frontier
    return chain

a = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_refix_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix1/partition6_refix_stitched.onnx')
b = ModelWrapper('/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/partition6_autosize_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_autosize1/partition6_autosize_postfifo.onnx')

pairs = [
    ("StreamingDataWidthConverter_rtl_10", "FMPadding_Pixel_hls_0"),
    ("FMPadding_Pixel_hls_0", "FMPadding_rtl_2"),
    ("FMPadding_rtl_2", "ConvolutionInputGenerator_rtl_2"),
    ("ConvolutionInputGenerator_rtl_2", "MVAU_rtl_8"),
    ("MVAU_rtl_8", "StreamingDataWidthConverter_rtl_13"),
    ("UpsampleNearestNeighbour_hls_0", "StreamingDataWidthConverter_rtl_10"),
    ("DuplicateStreams_hls_2", "UpsampleNearestNeighbour_hls_0"),
    ("DuplicateStreams_hls_2", "StreamingDataWidthConverter_rtl_10"),
]
for prod_name, cons_name in pairs:
    print(f"--- {prod_name} -> {cons_name} ---")
    try:
        print("  analytical:", fifo_depth_between(a, prod_name, cons_name))
    except KeyError as e:
        print("  analytical: NODE NOT FOUND", e)
    try:
        print("  autosize:  ", fifo_depth_between(b, prod_name, cons_name))
    except KeyError as e:
        print("  autosize: NODE NOT FOUND", e)
