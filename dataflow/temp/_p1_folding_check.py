import json, sys, warnings
warnings.filterwarnings("ignore")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
B = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/"
cfg = json.load(open(B + "hawq_folding_config_partition1.json"))
print("top keys:", list(cfg.keys())[:10], "n", len(cfg))
for k, v in list(cfg.items())[:6]:
    print(" ", k, json.dumps(v)[:300])
m = ModelWrapper(B + "intermediate_models/supported_op_partitions/partition_1_postfifo_autosize.onnx")
print("--- landed (graph order) ---")
for n in m.graph.node:
    if n.op_type.startswith(("MVAU", "VVAU", "ConvolutionInput", "Thresholding")):
        op = getCustomOp(n)
        a = {x: op.get_nodeattr(x) for x in ("PE", "SIMD") if x in op.get_nodeattr_types()}
        extra = ""
        if n.op_type.startswith("MVAU"):
            extra = f"MW={op.get_nodeattr('MW')} MH={op.get_nodeattr('MH')} w={op.get_nodeattr('weightDataType')} cyc={op.get_exp_cycles()}"
        print(n.name.ljust(34), a, extra)
