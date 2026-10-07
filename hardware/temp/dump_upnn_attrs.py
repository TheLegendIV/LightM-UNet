import onnx
for idx in (5, 6):
    m = onnx.load(rf"c:\DEV\repos\LightM-UNet\hardware\temp\p5_p6_diag\partition_{idx}_prefifo_autosize.onnx")
    print(f"=== partition {idx} ===")
    for n in m.graph.node:
        if n.op_type.startswith("UpsampleNearestNeighbour"):
            attrs = {a.name: (a.i if a.type == onnx.AttributeProto.INT else a.s) for a in n.attribute}
            print(n.name, n.op_type, attrs)
