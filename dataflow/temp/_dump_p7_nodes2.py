import sys
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
base = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions"
m = ModelWrapper(base + "/partition_7.onnx")
keep = ("PE", "SIMD", "numInputVectors", "OFMDim", "IFMDim", "KernelDim", "Stride", "Padding", "NumChannels", "ImgDim", "ConvKernelDim", "IFMChannels", "OFMChannels", "InputDims", "PaddingBottom", "PaddingRight", "Dilation", "depthwise", "NumInputVectors", "inputDataType", "outputDataType")
for n in m.graph.node:
    if "FIFO" in n.op_type or "DataWidth" in n.op_type:
        continue
    attrs = {}
    for a in n.attribute:
        if a.name in keep:
            v = helper.get_attribute_value(a)
            attrs[a.name] = v.decode() if isinstance(v, bytes) else (list(v) if hasattr(v, "__len__") else v)
    outs = [(m.get_tensor_shape(o), str(m.get_tensor_datatype(o))) for o in n.output]
    print(n.op_type, n.name.replace("GenericPartition_7_", ""), outs, attrs, flush=True)
