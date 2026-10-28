from qonnx.core.modelwrapper import ModelWrapper
import sys

path = sys.argv[1]
m = ModelWrapper(path)
g = m.graph
for n in g.node:
    if "AddStreams" in n.op_type:
        print("ADD", n.name, "inputs=", list(n.input))
        for inp in n.input:
            prod = m.find_producer(inp)
            if prod is None:
                print("   ", inp, "-> (graph input, no producer)")
                continue
            print("   ", inp, "->", prod.op_type, prod.name)
            # walk one more hop back for context
            for pin in prod.input:
                pprod = m.find_producer(pin)
                if pprod is not None:
                    print("       <-", pprod.op_type, pprod.name)
