import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTPUT_DIR = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_20260918_144349"
m = ModelWrapper(OUTPUT_DIR + "/intermediate_models/dataflow_parent.onnx")
sdp_nodes = m.get_nodes_by_op_type("StreamingDataflowPartition")
print(f"{len(sdp_nodes)} SDP nodes")
for i, n in enumerate(sdp_nodes):
    inst = getCustomOp(n)
    print(i, n.name, "model=", inst.get_nodeattr("model"))
