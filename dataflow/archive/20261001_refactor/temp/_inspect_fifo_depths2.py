import sys
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
    try:
        width = inst.get_outstream_width()
    except Exception:
        width = None
    est_bits = depth * width if width else None
    rows.append((node.name, depth, width, est_bits))

rows.sort(key=lambda r: -(r[3] or 0))
total_bits = sum(r[3] for r in rows if r[3])
print(f"=== {onnx_path} ===")
print(f"Total FIFO nodes: {len(rows)}")
print(f"Total FIFO storage: {total_bits} bits = {total_bits/8/1024:.1f} KB "
      f"= {total_bits/36864:.1f} x RAMB36 (36Kb each) -- device has 312 RAMB36 total")
print()
print(f"{'name':30s} {'depth':>8s} {'width(bits)':>12s} {'total_bits':>12s} {'RAMB36eq':>9s}")
for name, depth, width, est_bits in rows[:30]:
    ramb = est_bits / 36864 if est_bits else 0
    print(f"{name:30s} {depth:8d} {str(width):>12s} {str(est_bits):>12s} {ramb:9.2f}")
