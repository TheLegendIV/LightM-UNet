import sys

sys.path.insert(0, ".")
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC

MODEL_FILE = sys.argv[1]
model = ModelWrapper(MODEL_FILE)

for n in model.graph.node:
    if n.op_type == "MaxPool":
        producer = model.find_producer(n.input[0])
        consumer = model.find_consumer(n.output[0])
        print(
            "BEFORE", n.name, "producer=", producer.name if producer else None,
            producer.op_type if producer else None,
            list(get_by_name(producer.attribute, "perm").ints) if producer and producer.op_type == "Transpose" else None,
            "consumer=", consumer.name if consumer else None, consumer.op_type if consumer else None,
        )

m2, changed = MakeMaxPoolNHWC().apply(model)
print("changed=", changed)
for n in m2.graph.node:
    if n.op_type in ("MaxPool", "MaxPoolNHWC"):
        print("AFTER", n.name, n.op_type, list(n.input), list(n.output))
