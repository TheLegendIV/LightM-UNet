"""For every DuplicateStreams -> AddStreams pair in a stitched partition ONNX, list each branch's nodes and FIFOs.

Usage: python3 fork_join_branches.py <stitched.onnx>
"""
import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])


def walk(tensor, stop_op):
    """Follow single-consumer chain from tensor until a node whose op_type starts with stop_op."""
    chain = []
    while True:
        c = m.find_consumer(tensor)
        if c is None:
            return chain, None
        if c.op_type.startswith(stop_op):
            return chain, c
        chain.append(c)
        tensor = c.output[0]


def describe(n):
    op = getCustomOp(n)
    if n.op_type.startswith("StreamingFIFO"):
        shp = op.get_nodeattr("folded_shape")
        return "FIFO %-22s depth=%-5d width=%-4d beats=%-7d" % (
            n.name, op.get_nodeattr("depth"), op.get_instream_width(), int(np.prod(shp[:-1])))
    return "     %s" % n.name


for dup in [n for n in m.graph.node if n.op_type.startswith("DuplicateStreams")]:
    print("=== %s" % dup.name)
    for k, out in enumerate(dup.output):
        chain, join = walk(out, "AddStreams")
        port = list(join.input).index(chain[-1].output[0]) if join is not None and chain else "?"
        print("  out%d -> %s (in%s): %d nodes, %d FIFOs, total FIFO depth %d" % (
            k, join.name if join else None, port, len(chain),
            sum(1 for c in chain if c.op_type.startswith("StreamingFIFO")),
            sum(getCustomOp(c).get_nodeattr("depth") for c in chain if c.op_type.startswith("StreamingFIFO"))))
        for c in chain:
            print("     ", describe(c))
