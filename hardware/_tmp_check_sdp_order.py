from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

OUTDIR = "finn_deployment_outputs/hawq_26_9_w24_ptq_8way_full_20260829_094221"
m = ModelWrapper(f"{OUTDIR}/intermediate_models/dataflow_parent.onnx")
sdp_nodes = m.get_nodes_by_op_type("StreamingDataflowPartition")
for i, n in enumerate(sdp_nodes):
    inst = getCustomOp(n)
    print(i, n.name, inst.get_nodeattr("model"))
