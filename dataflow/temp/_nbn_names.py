import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import node_by_node_check as n
P = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/partition_1_postfifo_autosize.onnx"
m = ModelWrapper(P)
tm = {t["stream"]: t for t in json.loads(Path("/tmp/nbn_p1/taps/taps.json").read_text())}
for node in m.graph.node:
    op = getCustomOp(node)
    b = n.bd_name(op, node, "GenericPartition_1")
    s = f"{b}_out_V"
    t = tm.get(s)
    d = op.get_nodeattr("code_gen_dir_ipgen")
    print(node.name[:34].ljust(34), node.op_type[:24].ljust(24), "bd=" + b[16:][:40].ljust(40), "nb=%s" % (t["nb"] if t else None), "dir=" + Path(d).name[-30:] if d else "")
