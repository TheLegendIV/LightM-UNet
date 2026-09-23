import onnx
p = "finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_20260922_011241/intermediate_models/supported_op_partitions/partition_0.onnx"
m = onnx.load(p)
print(sorted(set(n.op_type for n in m.graph.node)))
