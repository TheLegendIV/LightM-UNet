import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
from qonnx.core.modelwrapper import ModelWrapper

MODEL = ("/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/"
         "S12_256_analytical_namefix_20261006_195156/intermediate_models/"
         "supported_op_partitions/partition_5_prefifo_autosize.onnx")

model = ModelWrapper(MODEL)
target = None
for n in model.graph.node:
    if n.name == "MVAU_rtl_11":
        target = n
        break
print("target:", target.name, target.op_type, "inputs:", list(target.input))
cur = target
for i in range(6):
    p = model.find_producer(cur.input[0])
    if p is None:
        print(f"hop {i}: producer is None (input tensor {cur.input[0]} is a graph input/initializer)")
        break
    print(f"hop {i}: {p.name} ({p.op_type}) inputs={list(p.input)}")
    cur = p
