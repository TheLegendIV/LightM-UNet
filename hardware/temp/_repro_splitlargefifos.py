"""Isolated repro: does SplitLargeFIFOs() actually split a real
StreamingFIFO_rtl custom-op node with depth=294912?
"""
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs

depth = 294912
width_bits = 8
shape = [1, 1, 1, 8]

inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

fifo_node = helper.make_node(
    "StreamingFIFO_rtl",
    ["inp"],
    ["outp"],
    domain="finn.custom_op.fpgadataflow.rtl",
    backend="fpgadataflow",
    depth=depth,
    folded_shape=shape,
    normal_shape=shape,
    dataType="UINT8",
    impl_style="rtl",
    ram_style="auto",
    name="StreamingFIFO_rtl_repro",
)

graph = helper.make_graph([fifo_node], "repro", [inp], [outp])
model = ModelWrapper(helper.make_model(graph, producer_name="repro"))
model.set_tensor_datatype("inp", DataType["UINT8"])
model.set_tensor_datatype("outp", DataType["UINT8"])
model = model.transform(GiveUniqueNodeNames())

print("BEFORE split:")
for n in model.graph.node:
    print(" ", n.op_type, n.name)

model2 = model.transform(SplitLargeFIFOs())

print("AFTER split:")
for n in model2.graph.node:
    from qonnx.custom_op.registry import getCustomOp

    inst = getCustomOp(n)
    print(" ", n.op_type, n.name, "depth=", inst.get_nodeattr("depth"), "impl_style=", inst.get_nodeattr("impl_style"))
