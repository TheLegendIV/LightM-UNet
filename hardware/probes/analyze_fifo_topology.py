"""Analysis helper (not a build step): for a built partition ONNX that has
already been through step_set_fifo_depths (StreamingFIFO_rtl nodes present
with depth/impl_style set), list all FIFOs sorted by depth descending, along
with each FIFO's immediate producer/consumer node names+op_types so large
FIFOs can be attributed to a specific fork (DuplicateStreams -> ... -> FIFO)
or join (FIFO -> ... -> AddStreams/StreamingConcat) point in the graph,
instead of just a bare "StreamingFIFO_rtl_N" name.

Usage (inside the FINN container):
    python3 analyze_fifo_topology.py <partition_onnx_path>
"""
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

FORK_OP_TYPES = {"DuplicateStreams_hls", "DuplicateStreams_rtl"}
JOIN_OP_TYPES = {"AddStreams_hls", "AddStreams_rtl", "StreamingConcat_hls", "StreamingConcat_rtl"}


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_fifo_topology.py <partition_onnx_path>")
        sys.exit(1)
    model = ModelWrapper(sys.argv[1])

    fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    rows = []
    for node in fifo_nodes:
        inst = getCustomOp(node)
        depth = inst.get_nodeattr("depth")
        impl_style = inst.get_nodeattr("impl_style")

        producer = model.find_producer(node.input[0])
        consumer_list = model.find_consumers(node.output[0])

        prod_desc = f"{producer.name} ({producer.op_type})" if producer is not None else "<GRAPH INPUT>"
        cons_desc = ", ".join(f"{c.name} ({c.op_type})" for c in consumer_list) if consumer_list else "<GRAPH OUTPUT>"

        # fork: the FIFO's producer itself has >1 consumer (i.e. the tensor
        # feeding this FIFO is one branch of a fork), or producer op_type is
        # a duplicate/split node.
        is_fork = False
        if producer is not None:
            prod_out_consumers = model.find_consumers(producer.output[0])
            is_fork = len(prod_out_consumers) > 1 or producer.op_type in FORK_OP_TYPES
        # join: this FIFO directly feeds an Add/Concat-type node.
        is_join = any(c.op_type in JOIN_OP_TYPES for c in consumer_list)

        tag = []
        if is_fork:
            tag.append("FORK-SOURCE")
        if is_join:
            tag.append("JOIN-TARGET")
        tag_str = "/".join(tag) if tag else "-"

        rows.append((depth, node.name, impl_style, prod_desc, cons_desc, tag_str))

    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"{'depth':>8}  {'impl':>6}  {'tag':<20}  {'fifo_name':<20}  producer -> consumer(s)")
    for depth, name, impl_style, prod_desc, cons_desc, tag_str in rows:
        print(f"{depth:>8}  {impl_style:>6}  {tag_str:<20}  {name:<20}  {prod_desc}  ->  {cons_desc}")


if __name__ == "__main__":
    main()
