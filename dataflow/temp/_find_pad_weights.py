import glob, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import onnx
from onnx import numpy_helper
PRE = "finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models/"
files = sorted(glob.glob(PRE + "*.onnx"), key=os.path.getmtime)
for f in files:
    m = onnx.load(f)
    inits = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    hits = []
    for name, ini in inits.items():
        a = numpy_helper.to_array(ini)
        if a.ndim >= 2 and a.size >= 32 and a.size <= 2048 and len(np.unique(a)) <= 2 and (np.unique(a) == 0).any() and np.abs(a).max() > 0:
            u = [n for n in cons.get(name, [])]
            hits.append((name, tuple(a.shape), np.unique(a).tolist(), [(n.op_type, n.name) for n in u]))
    print(os.path.basename(f), len(m.graph.node), "nodes;", len(hits), "binary-valued 2D+ initializers")
    for h in hits[:8]:
        print("   ", h)
