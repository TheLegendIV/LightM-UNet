import sys

import onnx

d = sys.argv[1]
m = onnx.load(f"{d}/intermediate_models/supported_op_partitions/partition_7_postfifo_autosize.onnx")
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        continue
    a = {}
    for at in n.attribute:
        if at.name in ("PE", "SIMD", "MH", "MW", "numInputVectors", "IFMDim", "OFMDim", "IFMChannels", "NumChannels", "ImgDim", "cycles_estimate", "NumInputVectors", "inFIFODepths", "Padding", "ConvKernelDim", "Stride", "parallel_window", "numOutputVectors"):
            a[at.name] = list(at.ints) if at.ints else (at.i if at.type == 2 else at.s.decode())
    print(n.name, a)
