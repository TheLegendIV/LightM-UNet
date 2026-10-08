import sys
import collections
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

PRE = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_analytical_v2_ft15ep_preamble_20261008_032400"
m = ModelWrapper(PRE + "/intermediate_models/assign_stage_partition_ids_8way.onnx")
dt = collections.Counter()
rows = []
for n in m.graph.node:
    if n.op_type.startswith(("MVAU", "VVAU")):
        inst = getCustomOp(n)
        w = m.get_initializer(n.input[1])
        wdt = str(inst.get_weight_datatype())
        dt[wdt] += 1
        rows.append((n.name, wdt, None if w is None else (w.shape, int(len(np.unique(w))), float(w.min()), float(w.max()), float(np.std(w)))))
print("weight dtype histogram:", dict(dt))
for r in rows:
    print(r)
