import json
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/thresh_initblock_simd9_rtl_bd_isolation_20261005_020424/step02_after_specialize.onnx") if False else None
import glob
cands = sorted(glob.glob("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/thresh_initblock_simd9_rtl_bd_isolation_*/step02_after_specialize.onnx"))
print(cands)
m = ModelWrapper(cands[-1])
for n in m.graph.node:
    if n.op_type == "MVAU_rtl":
        op = getCustomOp(n)
        for attr in ["MW", "MH", "PE", "SIMD", "inputDataType", "weightDataType", "outputDataType", "accDataType", "resType"]:
            try:
                print(attr, "=", op.get_nodeattr(attr))
            except Exception as e:
                print(attr, "ERR", e)
