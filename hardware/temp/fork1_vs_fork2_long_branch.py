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
        return "NO_PRODUCER"
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
    return found or "NOT_FOUND"


analytical = ModelWrapper(sys.argv[1])
autosize = ModelWrapper(sys.argv[2])

# long-branch internal edges, fork1 (up4-equiv) vs fork2 (up5)
edge_pairs = [
    ("fork1 long: MVAU_3->Thresh_7",      "MVAU_rtl_3", "Thresholding_rtl_7"),
    ("fork1 long: Thresh_7->FMPad_1",     "Thresholding_rtl_7", "FMPadding_rtl_1"),
    ("fork1 long: FMPad_1->ConvGen_1",    "FMPadding_rtl_1", "ConvolutionInputGenerator_rtl_1"),
    ("fork1 long: ConvGen_1->MVAU_4",     "ConvolutionInputGenerator_rtl_1", "MVAU_rtl_4"),
    ("fork1 long: MVAU_4->Thresh_8",      "MVAU_rtl_4", "Thresholding_rtl_8"),
    ("fork1 long: Thresh_8->MVAU_5",      "Thresholding_rtl_8", "MVAU_rtl_5"),
    ("fork1 long: MVAU_5->Thresh_9",      "MVAU_rtl_5", "Thresholding_rtl_9"),

    ("fork2 long: MVAU_7->Thresh_11",     "MVAU_rtl_7", "Thresholding_rtl_11"),
    ("fork2 long: Thresh_11->FMPadPix_0", "Thresholding_rtl_11", "FMPadding_Pixel_hls_0"),
    ("fork2 long: FMPadPix_0->FMPad_2",   "FMPadding_Pixel_hls_0", "FMPadding_rtl_2"),
    ("fork2 long: FMPad_2->ConvGen_2",    "FMPadding_rtl_2", "ConvolutionInputGenerator_rtl_2"),
    ("fork2 long: ConvGen_2->MVAU_8",     "ConvolutionInputGenerator_rtl_2", "MVAU_rtl_8"),
    ("fork2 long: MVAU_8->Thresh_14",     "MVAU_rtl_8", "Thresholding_rtl_14"),
    ("fork2 long: Thresh_14->MVAU_9",     "Thresholding_rtl_14", "MVAU_rtl_9"),
    ("fork2 long: MVAU_9->Thresh_15",     "MVAU_rtl_9", "Thresholding_rtl_15"),
]

for label, prod, cons in edge_pairs:
    a = fifo_depth_between(analytical, prod, cons)
    b = fifo_depth_between(autosize, prod, cons)
    print(f"{label:35s} analytical={a!s:30s} autosize={b!s:30s}")
