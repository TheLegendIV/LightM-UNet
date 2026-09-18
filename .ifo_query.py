import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918"
import os
parent_model = ModelWrapper(os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx"))
sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
node = sdp_nodes[2]
model_path = getCustomOp(node).get_nodeattr("model")
part_model = ModelWrapper(model_path)
print("model_path:", model_path)
print("vivado_stitch_proj:", part_model.get_metadata_prop("vivado_stitch_proj"))
print("wrapper_filename:", part_model.get_metadata_prop("wrapper_filename"))
fifo_nodes = part_model.get_nodes_by_op_type("StreamingFIFO_rtl")
for n in fifo_nodes:
    inst = getCustomOp(n)
    depth = inst.get_nodeattr("depth")
    if depth > 64:
        print(n.name, "depth=", depth, "ram_style=", inst.get_nodeattr("ram_style"))
