"""Dump StreamingFIFO_rtl depth + real LUT/FF/SRL cost per node, joined
against the same hier reports used for the main 8-way build's MVAU/VVAU/
Thresholding/SWU extended dataset -- answers "are any FIFOs deep enough
that a BRAM/URAM-backed impl_style=vivado FIFO would be worth it".
Run INSIDE the FINN container (HOME=/tmp/home_dir). Usage:
    HOME=/tmp/home_dir python3 dump_fifo_depths.py <stitched/partition.onnx> <out.json>
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

onnx_path, out_path = sys.argv[1], sys.argv[2]
model = ModelWrapper(onnx_path)

rows = []
for node in model.graph.node:
    if "StreamingFIFO" not in node.op_type:
        continue
    inst = getCustomOp(node)
    attrs = {}
    for key in ("depth", "impl_style", "ram_style", "dataType", "folded_shape"):
        try:
            attrs[key] = inst.get_nodeattr(key)
        except Exception:
            pass
    rows.append({"node_name": node.name, "op_type": node.op_type, "attrs": attrs})

with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"wrote {len(rows)} FIFO nodes to {out_path}")
