"""Host-runnable (no qonnx needed) clone of hardware/utils/dump_node_attrs_all.py.

Reads the same op_type/nodeattr superset directly off the ONNX protobuf
AttributeProto fields instead of going through qonnx's get_nodeattr() (which
applies per-op-type default values) -- fine here since every node of interest
has already been through FINN's real build pipeline and so always carries
its own explicit attribute values (no defaults needed). Lets this run with
plain `pip install onnx` on the Windows host, no container round-trip.

    python dump_node_attrs_local.py <model.onnx> <out.json>
"""
import json
import sys

import onnx

OP_TYPES = (
    "MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl",
    "Thresholding_hls", "Thresholding_rtl",
    "ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl",
    "StreamingFIFO_hls", "StreamingFIFO_rtl",
    "AddStreams_hls", "DuplicateStreams_hls", "StreamingConcat_hls",
    "UpsampleNearestNeighbour_hls", "StreamingMaxPool_hls",
    "StreamingDataWidthConverter_hls", "StreamingDataWidthConverter_rtl",
    "FMPadding_hls", "FMPadding_rtl", "FMPadding_Pixel_hls",
)


def _attr_value(a):
    if a.type == onnx.AttributeProto.INT:
        return a.i
    if a.type == onnx.AttributeProto.FLOAT:
        return a.f
    if a.type == onnx.AttributeProto.STRING:
        return a.s.decode()
    if a.type == onnx.AttributeProto.INTS:
        return list(a.ints)
    if a.type == onnx.AttributeProto.FLOATS:
        return list(a.floats)
    if a.type == onnx.AttributeProto.STRINGS:
        return [s.decode() for s in a.strings]
    return None


def main():
    onnx_path, out_path = sys.argv[1], sys.argv[2]
    model = onnx.load(onnx_path)

    rows = []
    for node in model.graph.node:
        if node.op_type not in OP_TYPES:
            continue
        attrs = {a.name: _attr_value(a) for a in node.attribute}
        rows.append({"node_name": node.name, "op_type": node.op_type, "attrs": attrs})

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(rows)} nodes to {out_path}")


if __name__ == "__main__":
    main()
