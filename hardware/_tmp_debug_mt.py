import sys, traceback
sys.path.insert(0, ".")
from finn_export_probe_s12_context_common import build_and_estimate, export_onnx
import onnx

try:
    model, est = build_and_estimate(False)
    path = export_onnx(model, "probe_s12_context_dense_int4_DEBUG")
    m = onnx.load(str(path))
    mt = [n for n in m.graph.node if n.op_type == "MultiThreshold"]
    with open("debug_mt_out.txt", "w") as f:
        f.write(f"num MultiThreshold nodes: {len(mt)}\n")
        for i, n in enumerate(mt):
            f.write(f"{i} {n.name} inputs: {list(n.input)} outputs: {list(n.output)}\n")
    print("WROTE debug_mt_out.txt OK")
except Exception:
    with open("debug_mt_out.txt", "w") as f:
        f.write(traceback.format_exc())
    print("WROTE debug_mt_out.txt EXCEPTION")
