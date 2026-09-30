"""Extended per-node attrs dumper for the mvau_variant_matrix probes:
unlike hardware/dump_node_attrs.py (MVAU/VVAU only), this ALSO captures
Thresholding_hls/_rtl (standalone threshold, relevant for the noact1/
unfused combos) and ConvolutionInputGenerator_hls/_rtl (SWU -- sliding
window unit feeding conv-like MVAU/VVAU nodes), so the calibration
dataset can account for the FULL per-node hardware footprint, not just
MVAU/VVAU. Run INSIDE the FINN container (HOME=/tmp/home_dir, needs
qonnx importable). Usage:
    HOME=/tmp/home_dir python3 dump_node_attrs_ext.py <stitched_ip.onnx> <out.json>
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

onnx_path, out_path = sys.argv[1], sys.argv[2]
model = ModelWrapper(onnx_path)

OP_TYPES = (
    "MVAU_hls", "VVAU_hls", "MVAU_rtl", "VVAU_rtl",
    "Thresholding_hls", "Thresholding_rtl",
    "ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl",
)

# union of every nodeattr key we might care about across all these op types
ATTR_KEYS = (
    "MH", "MW", "PE", "SIMD", "Channels", "Kernel",
    "weightDataType", "inputDataType", "outputDataType", "accDataType",
    "resType", "ram_style", "mem_mode", "runtime_writeable_weights",
    "NumChannels", "ConvKernelDim", "IFMChannels", "IFMDim", "OFMDim",
    "Stride", "Dilation", "SIMD1", "numSteps", "numInputVectors",
    "depthwise", "parallel_window", "distributed_state",
    "depth_trigger_bram", "depth_trigger_uram",
)

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

with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"wrote {len(rows)} nodes to {out_path}")
