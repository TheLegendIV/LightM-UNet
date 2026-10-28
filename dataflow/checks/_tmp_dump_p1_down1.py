import sys

import onnx

for label, d in (("INT8", sys.argv[1]), ("mixed68", sys.argv[2])):
    m = onnx.load(f"{d}/intermediate_models/supported_op_partitions/partition_1_postfifo_autosize.onnx")
    print("==", label)
    nodes = [n for n in m.graph.node]
    prod = {}
    for n in nodes:
        for o in n.output:
            prod[o] = n
    shown = 0
    for n in nodes:
        a = {}
        for at in n.attribute:
            if at.name in ("PE", "SIMD", "ram_style", "depth", "depth_trigger_bram", "parallel_window", "inWidth", "outWidth", "MW", "MH", "NumChannels", "ImgDim", "IFMDim", "IFMChannels", "ConvKernelDim", "Stride"):
                a[at.name] = list(at.ints) if at.ints else (at.i if at.type == 2 else at.s.decode())
        srcs = [prod[i].name for i in n.input if i in prod]
        print(f"  {n.name:36s} {a}  <- {srcs}")
        shown += 1
        if "AddStreams" in n.op_type:
            break
