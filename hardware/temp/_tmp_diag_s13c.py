import sys
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/pyverilator")
sys.path.insert(0, "/home/thelegendiv/finn/deps/finn-experimental")
from qonnx.core.modelwrapper import ModelWrapper

path = sys.argv[1] if len(sys.argv) > 1 else "step_enet_convert_to_hw.onnx"
model = ModelWrapper(path)
graph = model.graph

hw_domain_prefix = "finn.custom_op"
non_hw = []
for node in graph.node:
    if not node.domain.startswith(hw_domain_prefix):
        non_hw.append(node)

print(f"Total nodes: {len(graph.node)}, non-HW nodes: {len(non_hw)}")
from collections import Counter
print(Counter(n.op_type for n in non_hw))

print("\n--- Detailed non-HW node chains ---")
for node in non_hw:
    ins = list(node.input)
    outs = list(node.output)
    print(f"\n{node.op_type} name={node.name}")
    for i in ins:
        producer = model.find_producer(i)
        pinfo = f"{producer.op_type}({producer.name})" if producer else "INPUT/INIT"
        print(f"  in: {i}  <- {pinfo}")
    for o in outs:
        consumers = model.find_consumers(o)
        cinfo = [f"{c.op_type}({c.name})" for c in consumers] if consumers else ["OUTPUT"]
        print(f"  out: {o} -> {cinfo}")
