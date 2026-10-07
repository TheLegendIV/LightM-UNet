import sys
sys.path.insert(0, '.')
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
    producer_of, consumers_of = build_maps(model)
    prod_node = model.get_node_from_name(producer_name)
    if prod_node is None:
        return None
    found = []
    frontier = [prod_node]
    seen = {producer_name}
    for _ in range(50):
        new_frontier = []
        for node in frontier:
            for out_t in node.output:
                for nxt in consumers_of.get(out_t, []):
                    if nxt.name == consumer_name:
                        return found
                    if nxt.op_type.startswith("StreamingFIFO") and nxt.name not in seen:
                        depth = getCustomOp(nxt).get_nodeattr("depth")
                        found.append((nxt.name, depth))
                        seen.add(nxt.name)
                        new_frontier.append(nxt)
                    elif nxt.name not in seen:
                        seen.add(nxt.name)
                        new_frontier.append(nxt)
        frontier = new_frontier
        if not frontier:
            break
    return found


analytical = ModelWrapper(sys.argv[1])
autosize = ModelWrapper(sys.argv[2])

for prod, cons in [("Thresholding_rtl_6", "AddStreams_hls_1"), ("Thresholding_rtl_13", "AddStreams_hls_2")]:
    print(f"=== {prod} -> {cons} ===")
    print("  analytical:", fifo_depth_between(analytical, prod, cons))
    print("  autosize  :", fifo_depth_between(autosize, prod, cons))
