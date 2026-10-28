import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

D = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
pre = D + "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_20261009_220505/intermediate_models/assign_stage_partition_ids_8way.onnx"
fin = D + "S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261009_225105/intermediate_models/supported_op_partitions/partition_%s.onnx" % sys.argv[1]
KEYS = ["PE", "SIMD", "Dim", "Channels", "IFMChannels", "OFMChannels", "ConvKernelDim", "KernelDim", "IFMDim", "OFMDim", "Stride", "Dilation", "depthwise", "parallel_window", "Padding", "ImgSize", "NumChannels", "UpsampleFactor", "DimOut", "mem_mode", "MW", "MH", "Kernel", "Channels", "numInputVectors", "inputDataType", "outputDataType", "weightDataType", "ActVal", "numSteps"]

def show(path, tag, filt):
    m = ModelWrapper(path)
    print("=====", tag, path.split("/")[-1], "nodes", len(m.graph.node))
    for n in m.graph.node:
        if not filt(n):
            continue
        op = getCustomOp(n)
        d = {}
        for k in KEYS:
            try:
                v = op.get_nodeattr(k)
                d[k] = v
            except Exception:
                pass
        ws = []
        for i in n.input[1:]:
            w = m.get_initializer(i)
            if w is not None:
                ws.append((i, tuple(w.shape), float(np.sum(w)), float(np.min(w)), float(np.max(w))))
        print(n.op_type, n.name, d)
        for w in ws:
            print("    init", w)

if sys.argv[2] == "pre":
    show(pre, "PRE", lambda n: n.op_type in ("VVAU", "VectorVectorActivation", "VVAU_hls", "VVAU_rtl") or "Vector" in n.op_type or n.op_type in ("ConvolutionInputGenerator", "UpsampleNearestNeighbour", "FMPadding", "FMPadding_Pixel", "FMPadding_Batch", "Upsample", "Resize"))
else:
    show(fin, "FINAL", lambda n: n.op_type.startswith(("VVAU", "ConvolutionInputGenerator", "Upsample", "FMPadding")) or n.name in ("StreamingDataWidthConverter_hls_0",))
