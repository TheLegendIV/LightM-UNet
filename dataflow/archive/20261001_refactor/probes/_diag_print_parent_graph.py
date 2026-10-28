import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/tiny_2part_real_pipeline_test"
ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent.onnx")
if not os.path.isfile(ckpt):
    # fall back to the pre-partition HW-converted checkpoint if saved
    ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "supported_op_partitions")
print("loading", ckpt)


def is_hw(node):
    b = get_by_name(node.attribute, "backend")
    return b is not None and b.s.decode() == "fpgadataflow"


model = ModelWrapper(ckpt)
for idx, node in enumerate(model.graph.node):
    ins = list(node.input)
    outs = list(node.output)
    print(idx, node.op_type, node.name, "in=", ins, "out=", outs)
