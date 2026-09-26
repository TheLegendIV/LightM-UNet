from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

flat = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_trained_rtl_mvau_8way_full_v1_256x256_20260926_153428/intermediate_models/dataflow_parent.onnx"
m = ModelWrapper(flat)
sdp_nodes = m.get_nodes_by_op_type("StreamingDataflowPartition")
for n in sdp_nodes:
    inst = getCustomOp(n)
    fn = inst.get_nodeattr("model")
    try:
        pm = ModelWrapper(fn)
        ops = sorted(set(nd.op_type for nd in pm.graph.node))
        print(n.name, fn, ops)
    except Exception as e:
        print(n.name, fn, "ERROR", e)
