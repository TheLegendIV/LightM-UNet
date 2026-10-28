from qonnx.core.modelwrapper import ModelWrapper
OUT = "finn_deployment_outputs/12_dense_relu_nearest_conv_upsample_256_w8_16_v4_trained_rtl_mvau_8way_full_256x256_20260928_043156"
m = ModelWrapper(OUT + "/iodma_driver_20260928_125148/kernel_models/GenericPartition_0_with_input_iodma.onnx")
names = [n.name for n in m.graph.node]
print("total", len(names))
print("has prefix on old nodes:", all(n.startswith("GenericPartition_0_") or n == "IODMA_hls_0" for n in names))
print("dupes:", {n for n in names if names.count(n) > 1})
print("sample:", names[:3], "...", names[-3:])
