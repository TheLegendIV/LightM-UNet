import json
from qonnx.core.modelwrapper import ModelWrapper

PATH = (
    "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
    "partition1_refix2_S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix_refix2/"
    "partition1_refix2_stitched.onnx"
)

model = ModelWrapper(PATH)
graph = model.graph


def producer(t):
    return model.find_producer(t)


def consumer(t):
    return model.find_consumer(t)


def walk_forward(node, max_hops=12):
    """Print a straight-line chain of consumers starting from `node`'s output 0."""
    chain = []
    n = node
    for _ in range(max_hops):
        if n is None:
            break
        chain.append(f"{n.name}({n.op_type})")
        outs = n.output
        if not outs:
            break
        c = consumer(outs[0])
        n = c
    return " -> ".join(chain)


# Find down2's Dup node (DuplicateStreams_hls_5 per prior force-report)
dup = None
for n in graph.node:
    if n.name == "DuplicateStreams_hls_5":
        dup = n
        break
print("Dup node:", dup.name if dup else None, dup.output if dup else None)

if dup is not None:
    for i, out in enumerate(dup.output):
        c = consumer(out)
        print(f"  Dup output[{i}] ({out}) -> consumer: {c.name}({c.op_type}) if c else None")
        print(f"    chain: {walk_forward(c, max_hops=14)}")

# Also specifically print producer/consumer of Thresholding_rtl_27 and the FIFO around it
for name in ["Thresholding_rtl_25", "Thresholding_rtl_26", "Thresholding_rtl_27", "MVAU_rtl_16", "MVAU_rtl_17", "MVAU_rtl_18"]:
    node = None
    for n in graph.node:
        if n.name == name:
            node = n
            break
    if node is None:
        print(name, "NOT FOUND")
        continue
    prod = producer(node.input[0])
    cons = consumer(node.output[0])
    print(f"{name}({node.op_type}): producer={prod.name if prod else None}({prod.op_type if prod else None}) "
          f"consumer={cons.name if cons else None}({cons.op_type if cons else None})")
