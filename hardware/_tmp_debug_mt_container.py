import sys
sys.path.insert(0, "/home/thelegendiv/finn/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/qonnx/src")
sys.path.insert(0, "/home/thelegendiv/finn/deps/brevitas/src")
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")

from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_steps import step_qonnx_to_finn
import finn_build_probe_s12_context_int4 as bm

model = ModelWrapper("/home/thelegendiv/finn/notebooks/enet/probe_s12_context_dense_int4.onnx")
cfg = bm.cfg_probe
model = step_qonnx_to_finn(model, cfg)
model = bm.step_probe_tidy(model, cfg)
model = bm.step_probe_streamline(model, cfg)

mt_nodes = model.get_nodes_by_op_type("MultiThreshold")
print("num MultiThreshold nodes:", len(mt_nodes))
for i, n in enumerate(mt_nodes):
    out_name = n.output[0]
    dt = model.get_tensor_datatype(out_name)
    print(i, n.name, "out_tensor:", out_name, "out_dtype:", dt, "signed:", dt.signed())
    # producing input node (what feeds the threshold in)
    inp_name = n.input[0]
    producer = model.find_producer(inp_name)
    print("    in_tensor:", inp_name, "produced_by:", producer.name if producer else None, producer.op_type if producer else None)
