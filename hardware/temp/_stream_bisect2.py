import sys, numpy as np
sys.argv = sys.argv[:1] + ["x"] + sys.argv[1:]
import finn_enet_build as fb
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oe
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.datatype import DataType
oe.sanitize_quant_values = lambda m, i, c, *a, **k: c
ref = np.load(sys.argv[2]); src = sys.argv[3]
x = ref["u"][0].astype(np.float32).reshape(1, 1, 256, 256)
pred = ref["pred"][0].reshape(256, 256)
def agree(m):
    m = ModelWrapper(m.model)
    for n in m.graph.node:
        for i, t in enumerate(n.input):
            if t == "":
                n.input[i] = f"{n.name}_e{i}"; m.set_initializer(n.input[i], np.zeros((0,), np.float32))
    m.set_tensor_datatype(m.graph.output[0].name, DataType["FLOAT32"])
    o = execute_onnx(m, {m.graph.input[0].name: x})[m.graph.output[0].name]
    return float((o.argmax(1).reshape(256, 256) == pred).mean())
orig = ModelWrapper.transform
state = {"last": None, "n": 0, "depth": 0}
def patched(self, trn, *a, **k):
    state["depth"] += 1
    try:
        out = orig(self, trn, *a, **k)
    finally:
        state["depth"] -= 1
    if state["depth"] > 0:
        return out
    state["n"] += 1
    name = type(trn).__name__
    if name in ("GiveUniqueNodeNames", "GiveReadableTensorNames", "RemoveUnusedTensors", "SortGraph", "InferDataTypes", "InferShapes", "InferDataLayouts"):
        return out
    try:
        v = agree(out)
    except Exception as e:
        v = "ERR " + str(e)[:80]
    print(f"  n={state['n']} {name} agreement={v}", flush=True) if state["n"] > 36 else None; (sys.exit(0) if state["n"] > 50 else None)
    last = state["last"]
    if last is None or isinstance(v, str) != isinstance(last, str) or (isinstance(v, float) and abs(v - last) > 0.003):
        print(f"#{state['n']} after {name}: agreement={v}", flush=True)
    state["last"] = v
    return out
ModelWrapper.transform = patched
m = ModelWrapper(src)
fb.step_enet_streamline(m, None)
print("DONE", flush=True)

