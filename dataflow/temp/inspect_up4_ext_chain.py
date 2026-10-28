import onnx

m = onnx.load(r"c:\DEV\repos\LightM-UNet\hardware\temp\p5_p6_diag\partition_5_prefifo_autosize.onnx")
graph = m.graph
by_output = {}
for n in graph.node:
    for o in n.output:
        by_output[o] = n
nodes_by_name = {n.name: n for n in graph.node}


def consumers_of(tensor_name):
    return [n for n in graph.node if tensor_name in n.input]


print("=== Chain from Thresholding_rtl_16 (up4.thr_r) forward ===")
cur = nodes_by_name.get("Thresholding_rtl_16")
seen = set()
for _ in range(20):
    if cur is None or cur.name in seen:
        break
    seen.add(cur.name)
    print(cur.name, cur.op_type)
    cons = consumers_of(cur.output[0])
    if not cons:
        break
    cur = cons[0]
