import json
import sys

import onnx

d = sys.argv[1]
m = onnx.load(f"{d}/intermediate_models/supported_op_partitions/partition_0_postfifo_autosize.onnx")
for n in m.graph.node:
    if n.op_type.startswith("Thresholding"):
        a = {}
        for at in n.attribute:
            if at.name in ("NumChannels", "PE", "numSteps", "inputDataType", "outputDataType", "depth_trigger_bram", "depth_trigger_uram", "ActVal", "runtime_writeable_weights"):
                a[at.name] = at.i if at.type == 2 else at.s.decode()
        prod = [p.name + ":" + p.op_type for p in m.graph.node if set(p.output) & set(n.input)]
        cons = [c.name + ":" + c.op_type for c in m.graph.node if set(c.input) & set(n.output)]
        print(json.dumps({"node": n.name, **a, "producer": prod, "consumer": cons}))
