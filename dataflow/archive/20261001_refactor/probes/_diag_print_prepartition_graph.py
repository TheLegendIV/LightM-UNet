import os
import sys

sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/tiny_2part_real_pipeline_test"
ckpt = os.path.join(OUTPUT_DIR, "intermediate_models", "step_probe_convert_to_hw.onnx")
print("loading", ckpt, "exists=", os.path.isfile(ckpt))
if not os.path.isfile(ckpt):
    # not saved under this name by our script -- fall back: rebuild is
    # cheap, so just re-run the pre-partition pipeline steps quickly
    sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
    import importlib
    m = importlib.import_module("finn_build_probe_tiny_2part_real_pipeline_test")
    model = ModelWrapper(m.MODEL_FILE)
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    model = model.transform(ConvertQONNXtoFINN())
    model = m.step_probe_tidy(model)
    model = m.step_probe_streamline(model)
    model = m.step_probe_convert_to_hw(model)
else:
    model = ModelWrapper(ckpt)


def is_hw(node):
    b = get_by_name(node.attribute, "backend")
    return b is not None and b.s.decode() == "fpgadataflow"


hw_idx = 0
for idx, node in enumerate(model.graph.node):
    marker = ""
    if is_hw(node):
        marker = f"[hw#{hw_idx}]"
        hw_idx += 1
    print(idx, marker, node.op_type, node.name, "in=", list(node.input), "out=", list(node.output))
