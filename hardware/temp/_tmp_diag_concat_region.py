"""One-off diagnostic: dump perm/shape/dtype details around the new
Concat-based initial block's downsample region. Not part of the pipeline
-- run manually, delete after use.
"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

MODEL_FILE = sys.argv[1]
model = ModelWrapper(MODEL_FILE)

names = sys.argv[2:] if len(sys.argv) > 2 else [
    "Transpose_2", "Transpose_3", "Transpose_4", "Transpose_5", "MaxPool_0", "Concat_0",
]
for name in names:
    matches = [x for x in model.graph.node if x.name == name]
    if not matches:
        print(name, "NOT FOUND")
        continue
    n = matches[0]
    perm = get_by_name(n.attribute, "perm")
    axis = get_by_name(n.attribute, "axis")
    print(f"{name} op_type={n.op_type} perm={list(perm.ints) if perm else None} "
          f"axis={axis.i if axis else None} in={list(n.input)} out={list(n.output)}")
    for t in list(n.input) + list(n.output):
        print(f"    {t} shape={model.get_tensor_shape(t)} dtype={model.get_tensor_datatype(t)}")
