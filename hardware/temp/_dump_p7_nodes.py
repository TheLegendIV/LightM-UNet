import sys
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
base = "finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions"
import glob
fs = sorted(glob.glob(base + "/*partition_7*.onnx") + glob.glob(base + "/*7*.onnx"))
print(fs[:10])
EOF = None
m = ModelWrapper(fs[0])
for n in m.graph.node:
    ins = [(i, m.get_tensor_shape(i)) for i in n.input if m.get_initializer(i) is None]
    outs = [(o, m.get_tensor_shape(o), str(m.get_tensor_datatype(o))) for o in n.output]
    attrs = {a.name: (helper.get_attribute_value(a) if a.name in ("PE", "SIMD", "numInputVectors", "OFMDim", "IFMDim", "KernelDim", "Stride", "Padding", "NumChannels", "ImgDim", "Kernel", "SIMD") else None) for a in n.attribute}
    attrs = {k: v for k, v in attrs.items() if v is not None}
    print(n.op_type, n.name, ins, outs, attrs)
