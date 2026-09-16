"""Dump MVAU/VVAU + ConvolutionInputGenerator (SWU) + Thresholding nodeattrs
for partitions 0,2,3,4,5,6,7 of the 12_separable_dense_relu_alpha025_trained_8way
build. Run INSIDE the FINN container.
"""
import json
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_separable_dense_relu_alpha025_trained_8way_full_20260916_010009"
PARENT_CKPT = f"{OUTPUT_DIR}/intermediate_models/dataflow_parent_built.onnx"
PARTITIONS = [0, 2, 3, 4, 5, 6, 7]

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

parent_model = ModelWrapper(PARENT_CKPT)
sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")

for n in PARTITIONS:
    sdp_node = sdp_nodes[n]
    model_path = getCustomOp(sdp_node).get_nodeattr("model")
    part_model = ModelWrapper(model_path)
    rows = []
    for node in part_model.graph.node:
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
    out_path = f"/tmp/attrs_ext_partition_{n}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"partition {n}: wrote {len(rows)} nodes to {out_path}")
