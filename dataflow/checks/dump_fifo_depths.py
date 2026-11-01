"""Dump the StreamingFIFO depths of per-partition *_postfifo_autosize.onnx checkpoints (FINN container).

Usage: python3 dump_fifo_depths.py <supported_op_partitions dir> <out.json> [partition ids ...]

Edges are keyed '<producer node>-><consumer node>' (node names are deterministic for a given graph, so two
builds of the same partition can be compared edge by edge). Sizes use the depth after FINN's pow2 rounding
for impl_style=vivado FIFOs (what the IP really gets) and the stream width from the FIFO's folded shape.
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def pow2_ceil(x):
    return 1 << max(0, (int(x) - 1).bit_length())


def dump_partition(path):
    m = ModelWrapper(path)
    fifos = []
    for n in m.graph.node:
        if not n.op_type.startswith("StreamingFIFO"):
            continue
        op = getCustomOp(n)
        prod = m.find_producer(n.input[0])
        cons = m.find_consumer(n.output[0])
        depth = int(op.get_nodeattr("depth"))
        style = op.get_nodeattr("impl_style")
        width = int(op.get_instream_width())
        real_depth = pow2_ceil(depth) if style == "vivado" else depth
        fifos.append({
            "name": n.name,
            "edge": f"{prod.name if prod else 'IN'}->{cons.name if cons else 'OUT'}",
            "depth": depth,
            "real_depth": real_depth,
            "impl_style": style,
            "width_bits": width,
            "kbit": real_depth * width / 1024.0,
        })
    return fifos


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    ids = [int(x) for x in sys.argv[3:]] or list(range(8))
    res = {}
    for i in ids:
        fifos = dump_partition(f"{src}/partition_{i}_postfifo_autosize.onnx")
        res[str(i)] = fifos
        tot = sum(f["kbit"] for f in fifos)
        print(f"partition {i}: {len(fifos)} FIFOs, {tot / 1024:.2f} Mbit, max depth {max(f['real_depth'] for f in fifos)}")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
