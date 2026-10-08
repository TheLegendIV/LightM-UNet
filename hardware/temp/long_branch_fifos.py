import sys
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
ids = [38, 40, 43, 44, 46, 49, 51, 53, 55, 56, 57, 58, 59, 60]
by = {n.name: n for n in m.graph.node}
print("fifo  depth  width_bits  beats/frame  ram_style  impl  producer -> consumer")
for i in ids:
    n = by.get("StreamingFIFO_rtl_%d" % i)
    if n is None:
        print(i, "missing")
        continue
    op = getCustomOp(n)
    shp = op.get_nodeattr("folded_shape")
    beats = int(np.prod(shp[:-1]))
    p = m.find_producer(n.input[0])
    c = m.find_consumer(n.output[0])
    print(i, op.get_nodeattr("depth"), op.get_instream_width(), beats,
          op.get_nodeattr("ram_style"), op.get_nodeattr("impl_style"),
          (p.name if p else "-"), "->", (c.name if c else "-"))
