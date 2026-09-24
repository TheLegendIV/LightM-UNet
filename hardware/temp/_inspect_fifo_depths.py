import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

onnx_path = sys.argv[1]
model = ModelWrapper(onnx_path)

rows = []
for node in model.graph.node:
    if not node.op_type.startswith("StreamingFIFO"):
        continue
    inst = getCustomOp(node)
    depth = inst.get_nodeattr("depth")
    ram_style = inst.get_nodeattr("ram_style") if "ram_style" in [a.name for a in node.attribute] else "?"
    try:
        shape = inst.get_folded_output_shape(0)
    except Exception:
        shape = None
    dtype = inst.get_output_datatype(0) if hasattr(inst, "get_output_datatype") else None
    bits = dtype.bitwidth() if dtype is not None else None
    stream_width = inst.get_nodeattr("StreamWidth") if "StreamWidth" in [a.name for a in node.attribute] else None
    # bytes estimate: depth * stream width (bits) / 8
    est_bits = None
    if stream_width is not None:
        est_bits = depth * stream_width
    rows.append((node.name, depth, ram_style, stream_width, est_bits, shape))

rows.sort(key=lambda r: -r[1])
total_est_bits = sum(r[4] for r in rows if r[4] is not None)
print(f"=== {onnx_path} ===")
print(f"Total FIFO nodes: {len(rows)}")
print(f"Total estimated FIFO storage bits: {total_est_bits} ({total_est_bits/8/1024:.1f} KB) -> in BRAM36 (36Kb) equivalents: {total_est_bits/36864:.1f}")
print()
print(f"{'name':45s} {'depth':>8s} {'ram_style':>10s} {'width':>6s} {'est_bits':>10s} {'shape'}")
for name, depth, ram_style, width, est_bits, shape in rows[:40]:
    print(f"{name:45s} {depth:8d} {str(ram_style):>10s} {str(width):>6s} {str(est_bits):>10s} {shape}")
