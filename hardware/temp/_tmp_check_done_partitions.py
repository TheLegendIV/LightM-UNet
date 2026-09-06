import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType

WEIGHT_OP_TYPES = ("MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl")
OUTDIR = "finn_deployment_outputs/hawq_26_9_w24_ptq_8way_full_20260829_094221"

for i in [0, 1, 4]:
    path = f"{OUTDIR}/intermediate_models/supported_op_partitions/partition_{i}.onnx"
    m = ModelWrapper(path)
    for node in m.graph.node:
        if node.op_type not in WEIGHT_OP_TYPES:
            continue
        inst = getCustomOp(node)
        wdt = inst.get_nodeattr("weightDataType")
        w = m.get_initializer(node.input[1])
        if w is None:
            continue
        dtype = DataType[wdt]
        bad = int((~np.vectorize(dtype.allowed)(w)).sum())
        status = "BAD" if bad else "ok"
        print(i, node.name, node.op_type, wdt, status, bad, w.min(), w.max())
