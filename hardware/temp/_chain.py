import sys, numpy as np
from qonnx.core.modelwrapper import ModelWrapper
for p in sys.argv[1:]:
    m = ModelWrapper(p); print("==", p.split("/")[-1])
    for n in list(m.graph.node)[-7:]:
        ins = []
        for i in n.input:
            v = m.get_initializer(i)
            ins.append(i if v is None else f"{i}{tuple(v.shape)}{np.round(v.ravel()[:5],4).tolist()} {m.get_tensor_datatype(i)}")
        print(n.op_type, n.name, ins, "->", list(n.output))
