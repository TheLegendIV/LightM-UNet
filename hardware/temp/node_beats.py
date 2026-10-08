import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
names = sys.argv[2:]
for n in m.graph.node:
    if n.name in names:
        inst = getCustomOp(n)
        try:
            ishape = inst.get_folded_input_shape()
            oshape = inst.get_folded_output_shape()
            print(n.name, "folded_in", ishape, "folded_out", oshape,
                  "in_beats", int(__import__("numpy").prod(ishape[:-1])),
                  "out_beats", int(__import__("numpy").prod(oshape[:-1])))
        except Exception as e:
            print(n.name, "ERR", e)
