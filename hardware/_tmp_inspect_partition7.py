import onnx
from onnx import numpy_helper
import numpy as np

path = r"C:\DEV\repos\LightM-UNet\hardware\builds\12_dense_relu_nearest_conv_upsample_256_v2\results\finn_deployment_output_v2_20260926_234329\intermediate_models\supported_op_partitions\partition_7_prefifo_autosize.onnx"
m = onnx.load(path)
g = m.graph
print("=== GRAPH INPUTS ===")
for i in g.input:
    print(i.name)
print("=== GRAPH OUTPUTS ===")
for o in g.output:
    print(o.name)

init_map = {init.name: init for init in g.initializer}

print("\n=== ALL NODES (op_type, name, inputs, outputs) ===")
for n in g.node:
    print(f"{n.op_type:30s} name={n.name:40s} inputs={list(n.input)} outputs={list(n.output)}")

print("\n=== LAST 10 NODES DETAIL ===")
for n in g.node[-10:]:
    print(f"--- {n.name} ({n.op_type}) ---")
    print(f"inputs: {list(n.input)}")
    print(f"outputs: {list(n.output)}")
    for attr in n.attribute:
        print(f"  attr: {attr.name} = {onnx.helper.get_attribute_value(attr) if attr.type != onnx.AttributeProto.TENSOR else '<tensor>'}")
    for inp in n.input:
        if inp in init_map:
            arr = numpy_helper.to_array(init_map[inp])
            print(f"  initializer {inp}: shape={arr.shape} dtype={arr.dtype} sample={arr.flatten()[:8]}")

print("\n=== Look for ChannelwiseOp nodes ===")
for n in g.node:
    if 'channelwise' in n.op_type.lower() or 'channelwise' in n.name.lower():
        print(f"--- {n.name} ({n.op_type}) ---")
        print(f"inputs: {list(n.input)}")
        print(f"outputs: {list(n.output)}")
        for attr in n.attribute:
            try:
                val = onnx.helper.get_attribute_value(attr)
            except Exception:
                val = '<err>'
            print(f"  attr: {attr.name} = {val}")
        for inp in n.input:
            if inp in init_map:
                arr = numpy_helper.to_array(init_map[inp])
                print(f"  initializer {inp}: shape={arr.shape} dtype={arr.dtype} values={arr.flatten()[:16]}")
            else:
                for n2 in g.node:
                    if inp in n2.output:
                        print(f"  produced by node: {n2.name} ({n2.op_type})")
