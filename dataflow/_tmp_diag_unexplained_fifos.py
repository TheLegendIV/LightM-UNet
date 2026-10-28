"""Throwaway diagnostic: load the already-saved partition_6_shallow_fifos.onnx and trace the REAL
producer/consumer chain around StreamingFIFO_rtl_42 and StreamingFIFO_rtl_45 (the 2 FIFOs that
step_force_fifo_depths_from_milp's match report could not explain via any known role pattern), to
identify what real topology they sit on.
"""
from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper("/tmp/_tmp_probe_partition_fifo_out/partition_6_shallow_fifos.onnx")


def describe(n):
    if n is None:
        return "None"
    return f"{n.name} ({n.op_type})  in={list(n.input)}  out={list(n.output)}"


def walk_back(tensor, steps=4):
    chain = []
    t = tensor
    for _ in range(steps):
        p = m.find_producer(t)
        chain.append(describe(p))
        if p is None:
            break
        t = p.input[0]
    return chain


def walk_fwd(tensor, steps=4):
    chain = []
    t = tensor
    for _ in range(steps):
        c = m.find_consumer(t)
        chain.append(describe(c))
        if c is None:
            break
        t = c.output[0]
    return chain


for target in ("StreamingFIFO_rtl_42", "StreamingFIFO_rtl_45"):
    n = next((x for x in m.graph.node if x.name == target), None)
    print(f"\n=== {target} ===")
    if n is None:
        print("  NOT FOUND in this graph")
        continue
    print(" node:", describe(n))
    print(" input tensor:", n.input[0])
    print(" output tensor:", n.output[0])
    prod = m.find_producer(n.input[0])
    print(" immediate producer:", describe(prod))
    cons = m.find_consumer(n.output[0])
    print(" immediate consumer:", describe(cons))
    print(" backward chain from input (producer side, up to 5 hops):")
    for s in walk_back(n.input[0], 5):
        print("   ", s)
    print(" forward chain from output (consumer side, up to 5 hops):")
    for s in walk_fwd(n.output[0], 5):
        print("   ", s)
    # is the input/output tensor a genuine graph boundary (top-level input/output)?
    print(" is graph input:", n.input[0] in [i.name for i in m.graph.input])
    print(" is graph output:", n.output[0] in [o.name for o in m.graph.output])
