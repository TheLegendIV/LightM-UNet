import sys
from qonnx.core.modelwrapper import ModelWrapper

fn = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600/intermediate_models/assign_stage_partition_ids_8way.onnx"
m = ModelWrapper(fn)
sdp_nodes = m.get_nodes_by_op_type("StreamingDataflowPartition")
for n in sdp_nodes:
    print(repr(n.name))
