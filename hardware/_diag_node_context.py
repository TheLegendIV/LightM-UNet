from qonnx.core.modelwrapper import ModelWrapper
import sys

path = sys.argv[1]
target = sys.argv[2]
m = ModelWrapper(path)
g = m.graph
node = next(n for n in g.node if n.name == target)
print("NODE", node.name, node.op_type, "inputs=", list(node.input))
for inp in node.input:
    prod = m.find_producer(inp)
    if prod is None:
        print("  <-", inp, "(graph input)")
        continue
    print("  <-", prod.op_type, prod.name, "inputs=", list(prod.input))
    for pin in prod.input:
        pprod = m.find_producer(pin)
        if pprod is not None:
            print("      <<-", pprod.op_type, pprod.name)
# also find consumers of this node's output
for out in node.output:
    consumers = m.find_consumers(out)
    for c in consumers:
        print("  ->", c.op_type, c.name)
