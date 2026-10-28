import warnings
warnings.filterwarnings("ignore")
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

D = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/S12_dense_256_u4_u8in_int6_fps250_lat200_milpfold_8way_20261008_182350/intermediate_models/supported_op_partitions/"
k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
m = ModelWrapper(D + f"partition_{k}.onnx")
for n in m.graph.node:
    if n.op_type == "StreamingFIFO_rtl":
        continue
    o = getCustomOp(n)
    attrs = {}
    for a in ("MW", "MH", "SIMD", "PE", "NumChannels", "ImgDim", "IFMDim", "OFMDim", "Kernel", "KernelSize", "Stride", "inputDataType", "outputDataType",
              "weightDataType", "accDataType", "ActVal", "numInputVectors", "mem_mode", "noActivation", "ConvKernelDim", "IFMChannels", "OFMChannels", "Padding", "Padding_pad", "DepthWise"):
        try:
            v = o.get_nodeattr(a)
            attrs[a] = v
        except Exception:
            pass
    print(n.op_type, n.name, list(n.input)[:1], "->", list(n.output)[:1], attrs)
