import sys
from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper(sys.argv[1])
target = sys.argv[2]
nodes = {n.name: n for n in m.graph.node}


def producer(tensor):
    p = m.find_producer(tensor)
    return p


def trace(tensor, depth=0, maxdepth=60):
    chain = []
    while tensor is not None and len(chain) < maxdepth:
        p = producer(tensor)
        if p is None:
            chain.append("<graph input %s>" % tensor)
            break
        chain.append(p.name)
        if p.op_type.startswith("DuplicateStreams"):
            # report which fork output we came from
            idx = list(p.output).index(tensor)
            chain[-1] += "[out%d]" % idx
            break
        tensor = p.input[0]
    return chain


t = nodes[target]
for i, inp in enumerate(t.input):
    print("%s.in%d <- " % (target, i) + " <- ".join(trace(inp)))
