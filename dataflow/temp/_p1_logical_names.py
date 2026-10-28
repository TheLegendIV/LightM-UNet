import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
import finn_s12_build_steps as s
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
PRE = "finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934"
res = s.load_partition_logical_names(PRE, "/tmp/conv_order.json")
conv, pool = res[1]
print("p1 logical conv names:", len(conv))
B = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/"
m = ModelWrapper(B + "partition_1_postfifo_autosize.onnx")
w = [n for n in m.graph.node if n.op_type in s.WEIGHT_OP_TYPES]
print("p1 weight nodes:", len(w))
for i in range(max(len(w), len(conv))):
    n = w[i] if i < len(w) else None
    d = ""
    if n is not None:
        op = getCustomOp(n)
        d = f"{n.name:12s} MW={op.get_nodeattr('MW') if 'MW' in op.get_nodeattr_types() else '-'} MH={op.get_nodeattr('MH') if 'MH' in op.get_nodeattr_types() else '-'} w={op.get_nodeattr('weightDataType')}"
    print(i, (conv[i] if i < len(conv) else "-").ljust(24), d)
