import sys, warnings
warnings.filterwarnings("ignore")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
D = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/"
for k in range(8):
    m = ModelWrapper(D + f"partition_{k}_postfifo_autosize.onnx")
    for n in m.graph.node:
        if n.op_type.startswith(("MVAU", "VVAU")):
            op = getCustomOp(n)
            wdt = op.get_nodeattr("weightDataType")
            W = m.get_initializer(n.input[1])
            flag = "UNSIGNED" if not str(wdt).startswith("INT") and "BIPOLAR" not in str(wdt) else ""
            if flag or "-v" in sys.argv:
                print(k, n.name, n.op_type, "w", wdt, "in", op.get_nodeattr("inputDataType"), "W range", W.min(), W.max(), "MWxMH", op.get_nodeattr("MW") if n.op_type.startswith("MVAU") else "", op.get_nodeattr("MH") if n.op_type.startswith("MVAU") else "", flag)
