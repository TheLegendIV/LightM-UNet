import sys
sys.path.insert(0, "/home/thelegendiv/finn/notebooks/enet")
import onnx
from finn_export_probe_s12_context_common import build_and_estimate

model, est = build_and_estimate(False)

from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
import torch

out_path = "/tmp/probe_dbg_raw.onnx"
model.cpu().eval()
dummy = torch.randn(1, 1, 32, 32)
export_qonnx(model, export_path=out_path, input_t=dummy)
qonnx_cleanup(out_path, out_file=out_path)

m = onnx.load(out_path)
qnodes = [n for n in m.graph.node if n.op_type == "Quant"]
print("num Quant nodes:", len(qnodes))
for i, n in enumerate(qnodes):
    signed = None
    for a in n.attribute:
        if a.name == "signed":
            signed = a.i
    is_act = "_param" not in n.input[0]
    print(i, n.name, "input0:", n.input[0], "output:", n.output[0], "signed_attr:", signed, "ACT" if is_act else "weight/bias")
