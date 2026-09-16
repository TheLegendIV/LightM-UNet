"""Dump MVAU/VVAU + ConvolutionInputGenerator (SWU) + Thresholding nodeattrs
for the dense S12 context probe (int6, forced PE=MH/SIMD=1). Run INSIDE the
FINN container.
"""
import json
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

MODEL_PATH = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/probe_s12_context_dense_int6_pemh_simd1_20260916_170945/intermediate_models/step_create_stitched_ip.onnx"

OP_TYPES = (
    "MVAU_hls", "VVAU_hls", "MVAU_rtl", "VVAU_rtl",
    "ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl",
    "Thresholding_hls", "Thresholding_rtl",
)

ATTR_KEYS = (
    "MH", "MW", "PE", "SIMD", "Channels", "Kernel",
    "IFMChannels", "IFMDim", "OFMDim", "ConvKernelDim", "Stride", "Dilation",
    "depthwise", "parallel_window",
    "NumChannels", "numSteps", "ActVal",
    "weightDataType", "inputDataType", "outputDataType", "accDataType",
    "resType", "ram_style", "mem_mode", "runtime_writeable_weights",
)

model = ModelWrapper(MODEL_PATH)
rows = []
for node in model.graph.node:
    if node.op_type not in OP_TYPES:
        continue
    inst = getCustomOp(node)
    attrs = {}
    for key in ATTR_KEYS:
        try:
            attrs[key] = inst.get_nodeattr(key)
        except Exception:
            pass
    rows.append({"node_name": node.name, "op_type": node.op_type, "attrs": attrs})

out_path = "/tmp/attrs_ext_probe_dense_int6_pemh.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"wrote {len(rows)} nodes to {out_path}")
