import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

onnx_path = sys.argv[1]
targets = sys.argv[2].split(",")
model = ModelWrapper(onnx_path)

for t in targets:
    node = None
    for n in model.graph.node:
        if n.name == t:
            node = n
            break
    if node is None:
        print(t, "NOT FOUND")
        continue
    inst = getCustomOp(node)
    print(f"=== {t} ({node.op_type}) ===")
    for a in node.attribute:
        print(f"  {a.name} = {inst.get_nodeattr(a.name)}")
    try:
        print(f"  in_stream_width_bits = {inst.get_instream_width()}")
    except Exception as e:
        print(f"  in_stream_width_bits: ERR {e}")
    try:
        print(f"  out_stream_width_bits = {inst.get_outstream_width()}")
    except Exception as e:
        print(f"  out_stream_width_bits: ERR {e}")
    try:
        print(f"  folded_input_shape = {inst.get_folded_input_shape()}")
    except Exception as e:
        pass
    try:
        print(f"  folded_output_shape = {inst.get_folded_output_shape()}")
    except Exception as e:
        pass
    try:
        print(f"  normal_input_shape = {inst.get_normal_input_shape()}")
    except Exception as e:
        pass
    print()
