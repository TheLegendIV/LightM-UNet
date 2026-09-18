"""One-off: dump StreamingFIFO depths (from ONNX nodeattrs, not RTL) across all
8 partitions of the 12_dense_relu_warmstart150ep_alpha025 v2 8-way build, to a
CSV for inspection. Read-only, safe to run while OOC synth is in progress
(SynthOutOfContext never writes the ONNX back to disk).
"""
import csv
import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918"
PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")

parent_model = ModelWrapper(PARENT_CKPT)
sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
print(f"Found {len(sdp_nodes)} partitions")

FIFO_OP_TYPES = ["StreamingFIFO_rtl", "StreamingFIFO_hls", "StreamingFIFO"]

rows = []
for i, sdp_node in enumerate(sdp_nodes):
    model_path = getCustomOp(sdp_node).get_nodeattr("model")
    part_model = ModelWrapper(model_path)
    fifo_nodes = []
    for op_type in FIFO_OP_TYPES:
        fifo_nodes += part_model.get_nodes_by_op_type(op_type)
    for n in fifo_nodes:
        inst = getCustomOp(n)
        depth = inst.get_nodeattr("depth")
        try:
            ram_style = inst.get_nodeattr("ram_style")
        except Exception:
            ram_style = ""
        bitwidth = None
        dtype = ""
        try:
            dtype_str = str(inst.get_nodeattr("dataType"))
            dtype = dtype_str
            from qonnx.core.datatype import DataType
            bitwidth = DataType[dtype_str].bitwidth()
        except Exception:
            pass
        folded_shape = None
        folded_shape_str = ""
        try:
            folded_shape = inst.get_folded_output_shape()
            folded_shape_str = str(folded_shape)
        except Exception:
            pass
        elems_per_beat = 1
        if folded_shape:
            elems_per_beat = folded_shape[-1]  # per-cycle entry width, not full tensor size
        size_elems = elems_per_beat * depth  # total FIFO capacity in elements (entry_width * depth)
        size_bits = size_elems * bitwidth if bitwidth is not None else ""
        size_bytes = (size_bits / 8) if isinstance(size_bits, (int, float)) else ""
        rows.append({
            "partition": i,
            "node_name": n.name,
            "op_type": n.op_type,
            "depth": depth,
            "ram_style": ram_style,
            "dataType": dtype,
            "folded_shape": folded_shape_str,
            "elems_per_beat": elems_per_beat,
            "size_elems": size_elems,
            "bitwidth": bitwidth if bitwidth is not None else "",
            "size_bits": size_bits,
            "size_bytes": size_bytes,
        })
    print(f"partition {i}: {len(fifo_nodes)} FIFO nodes")

out_csv = "/tmp/fifo_depths.csv"
fieldnames = ["partition", "node_name", "op_type", "depth", "ram_style", "dataType",
              "folded_shape", "elems_per_beat", "size_elems", "bitwidth", "size_bits", "size_bytes"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} FIFO entries to {out_csv}")
