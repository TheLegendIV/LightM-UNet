import os
import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918"
PARENT_CKPT = os.path.join(OUTPUT_DIR, "intermediate_models", "dataflow_parent_built.onnx")
parent_model = ModelWrapper(PARENT_CKPT)
sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
for i, sdp_node in enumerate(sdp_nodes):
    model_path = getCustomOp(sdp_node).get_nodeattr("model")
    part_model = ModelWrapper(model_path)
    print(f"partition {i}: {part_model.get_metadata_prop('vivado_stitch_proj')}")
