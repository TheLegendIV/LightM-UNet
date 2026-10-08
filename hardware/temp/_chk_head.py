import sys, onnx, numpy as np
from onnx import numpy_helper as nh
for p in sys.argv[1:]:
    m = onnx.load(p)
    init = {i.name: nh.to_array(i) for i in m.graph.initializer}
    print("==", p.split("/")[-1])
    for n in m.graph.node:
        if n.name in ("Sub_0", "Mul_0"):
            print("  ", n.op_type, n.name, [(i, init[i].ravel()[:3]) for i in n.input if i in init])
    mts = [n for n in m.graph.node if n.op_type == "MultiThreshold"]
    if not mts:
        continue
    mt = mts[0]
    t = init.get(mt.input[1])
    print("   MT0 inputs", list(mt.input), "thr", None if t is None else (t.shape, t.min(), t.max()))
    print("   MT0 attrs", {a.name: (a.f if a.type == 1 else a.s) for a in mt.attribute})
