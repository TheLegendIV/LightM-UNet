"""Combined per-node attrs dumper: MVAU/VVAU, Thresholding, SWU
(ConvolutionInputGenerator) AND StreamingFIFO nodes in one pass -- superset
of hardware/dump_node_attrs.py (MVAU/VVAU only) and dump_fifo_depths.py
(FIFO only), used by build_node_resource_calibration_csv.py so a single
attrs JSON covers every node kind that needs joining against a real Vivado
hierarchical utilization report.

Run INSIDE the FINN container (needs qonnx importable):
    HOME=/tmp/home_dir python3 dump_node_attrs_all.py <partition.onnx> <out.json>
"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OP_TYPES = (
    "MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl",
    "Thresholding_hls", "Thresholding_rtl",
    "ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl",
    "StreamingFIFO_hls", "StreamingFIFO_rtl",
    # join/stream nodes (finn_milp.py's "extra_nodes": add/dup/concat/upsample/
    # maxpool) -- previously excluded here, so they never appeared in any
    # mvau_lut_calibration_dataset_*.csv despite being ~48% of this repo's own
    # modeled LUT budget for the S12-dense-256 v2 build (see _diagnostics.
    # extra_by_kind's residual_add/skip_quant/out_act/add/dup entries).
    "AddStreams_hls", "DuplicateStreams_hls", "StreamingConcat_hls",
    "UpsampleNearestNeighbour_hls", "StreamingMaxPool_hls",
)

# union of every nodeattr key any of the above op types might expose
ATTR_KEYS = (
    "MH", "MW", "PE", "SIMD", "Channels", "Kernel",
    "weightDataType", "inputDataType", "outputDataType", "accDataType",
    "resType", "ram_style", "mem_mode", "runtime_writeable_weights",
    "NumChannels", "numSteps",
    "IFMChannels", "IFMDim", "OFMDim", "ConvKernelDim", "Stride", "Dilation",
    "depthwise", "parallel_window",
    "depth", "impl_style", "dataType", "folded_shape",
)


def main():
    onnx_path, out_path = sys.argv[1], sys.argv[2]
    model = ModelWrapper(onnx_path)

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


if __name__ == "__main__":
    main()
