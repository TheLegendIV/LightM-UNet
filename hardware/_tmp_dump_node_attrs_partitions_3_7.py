"""Dump MVAU_hls/VVAU_hls nodeattrs for partitions 3-7 of the
12_separable_dense_relu_alpha025_trained_8way build (partitions 0/2 already
done, NOT regenerated here). Run INSIDE the FINN container.
"""
import json
import sys

sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_separable_dense_relu_alpha025_trained_8way_full_20260916_010009"
PARENT_CKPT = f"{OUTPUT_DIR}/intermediate_models/dataflow_parent_built.onnx"
PARTITIONS = [3, 4, 5, 6, 7]

ATTR_KEYS = (
    "MH", "MW", "PE", "SIMD", "Channels", "Kernel",
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
        if node.op_type not in ("MVAU_hls", "VVAU_hls"):
            continue
        inst = getCustomOp(node)
        attrs = {}
        for key in ATTR_KEYS:
            try:
                attrs[key] = inst.get_nodeattr(key)
            except Exception:
                pass
        rows.append({"node_name": node.name, "op_type": node.op_type, "attrs": attrs})
    out_path = f"/tmp/attrs_partition_{n}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"partition {n}: wrote {len(rows)} nodes to {out_path}")
