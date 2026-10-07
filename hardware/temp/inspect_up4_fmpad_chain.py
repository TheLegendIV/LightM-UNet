"""Inspect the real node-level producer/consumer chain around FMPadding_Pixel_hls in
partition 5's prefifo onnx, to confirm whether finn_s12_build_steps.py's FMPAD_OP_TYPES
op_type string mismatch (missing the _hls/_rtl suffix for FMPadding_Pixel) causes this
node to be silently treated as "no FMPadding" when walking back from the up-block's 3x3
conv (mvau_u), and to see where this node actually sits.
"""
import onnx

m = onnx.load(r"c:\DEV\repos\LightM-UNet\hardware\temp\p5_p6_diag\partition_5_prefifo_autosize.onnx")
graph = m.graph

by_output = {}
for n in graph.node:
    for o in n.output:
        by_output[o] = n

nodes_by_name = {n.name: n for n in graph.node}


def producer_of(tensor_name):
    return by_output.get(tensor_name)


def consumers_of(tensor_name):
    return [n for n in graph.node if tensor_name in n.input]


target = nodes_by_name.get("FMPadding_Pixel_hls_0")
print("=== FMPadding_Pixel_hls_0 ===")
print("op_type:", target.op_type)
print("inputs:", list(target.input))
print("outputs:", list(target.output))
p = producer_of(target.input[0])
print("producer:", p.name, p.op_type)
for c in consumers_of(target.output[0]):
    print("consumer:", c.name, c.op_type)

print("\n=== FMPadding_rtl_3 ===")
target2 = nodes_by_name.get("FMPadding_rtl_3")
print("inputs:", list(target2.input))
print("outputs:", list(target2.output))
p2 = producer_of(target2.input[0])
print("producer:", p2.name, p2.op_type)
for c in consumers_of(target2.output[0]):
    print("consumer:", c.name, c.op_type)

print("\n=== Full main-path chain from UpsampleNearestNeighbour_hls_0 forward ===")
cur = nodes_by_name.get("UpsampleNearestNeighbour_hls_0")
seen = set()
for _ in range(15):
    if cur is None or cur.name in seen:
        break
    seen.add(cur.name)
    print(cur.name, cur.op_type, "->", list(cur.output))
    cons = consumers_of(cur.output[0])
    if not cons:
        break
    cur = cons[0]
