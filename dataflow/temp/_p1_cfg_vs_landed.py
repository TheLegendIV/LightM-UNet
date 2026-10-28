import json, warnings
warnings.filterwarnings("ignore")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
B = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/"
cfg = json.load(open(B + "hawq_folding_config_partition1.json"))
m = ModelWrapper(B + "intermediate_models/supported_op_partitions/partition_1_postfifo_autosize.onnx")
names = {n.name: n for n in m.graph.node}
bad = 0
for k, v in cfg.items():
    if k == "Defaults":
        continue
    if k not in names:
        print("config key not in landed ONNX:", k, v); bad += 1; continue
    op = getCustomOp(names[k])
    for a, want in v.items():
        if a in op.get_nodeattr_types() and op.get_nodeattr(a) != want:
            print("MISMATCH", k, a, "cfg", want, "landed", op.get_nodeattr(a)); bad += 1
print("config entries", len(cfg) - 1, "mismatches", bad)
unc = [n.name for n in m.graph.node if n.op_type.startswith(("MVAU", "VVAU", "Thresholding", "ConvolutionInput", "FMPadding")) and n.name not in cfg]
print("landed compute nodes without a config entry:", unc)
