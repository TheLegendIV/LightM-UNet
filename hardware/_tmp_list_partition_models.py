import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

parent = ModelWrapper(
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "stitched_ip_partitioned8way_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260820_101224/"
    "intermediate_models/step_build_all_partitions_capped.onnx"
)
sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
print("num partitions:", len(sdp_nodes))
for node in sdp_nodes:
    sdp_inst = getCustomOp(node)
    model_fn = sdp_inst.get_nodeattr("model")
    km = ModelWrapper(model_fn)
    proj = km.get_metadata_prop("vivado_stitch_proj")
    wrapper = km.get_metadata_prop("wrapper_filename")
    print(node.name, "|", model_fn, "|", proj, "|", wrapper)
