"""Dump per-node (MVAU_hls/VVAU_hls) nodeattrs from a built partition ONNX,
for building a hardware-calibration CSV row set. Run INSIDE the FINN
container (needs qonnx importable). Usage:
    python3 _tmp_dump_node_attrs.py <partition.onnx> <out.json>
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

onnx_path, out_path = sys.argv[1], sys.argv[2]
model = ModelWrapper(onnx_path)

rows = []
for node in model.graph.node:
    if node.op_type not in ("MVAU_hls", "VVAU_hls"):
        continue
    inst = getCustomOp(node)
    attrs = {}
    for key in (
        "MH", "MW", "PE", "SIMD", "Channels", "Kernel",
        "weightDataType", "inputDataType", "outputDataType", "accDataType",
        "resType", "ram_style", "mem_mode", "runtime_writeable_weights",
    ):
        try:
            attrs[key] = inst.get_nodeattr(key)
        except Exception:
            pass
    rows.append({"node_name": node.name, "op_type": node.op_type, "attrs": attrs})

with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"wrote {len(rows)} nodes to {out_path}")
