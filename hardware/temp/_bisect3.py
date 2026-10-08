import sys, numpy as np
exec(open("_bisect.py").read().split("ca = execute_onnx")[0].replace("ref = np.load(sys.argv[1]); a, b = load(sys.argv[2]), load(sys.argv[3])", "ref = np.load(sys.argv[1]); a, b = load(sys.argv[2]), load(sys.argv[3])"))
x = ref["u"][0].astype(np.float32).reshape(1, 1, 256, 256)
ca = execute_onnx(a, {a.graph.input[0].name: x}, return_full_exec_context=True)
cb = execute_onnx(b, {b.graph.input[0].name: x}, return_full_exec_context=True)
mts_a = [(n.name, ca[n.output[0]]) for n in a.graph.node if n.op_type == "MultiThreshold"]
def variants(t):
    t = np.asarray(t); yield t
    if t.ndim == 4:
        yield t.transpose(0, 3, 1, 2); yield t.transpose(0, 2, 3, 1)
bad = 0; ok = 0
for n in b.graph.node:
    if n.op_type != "MultiThreshold":
        continue
    o = np.asarray(cb[n.output[0]]); best = None
    for na, ta in mts_a:
        for v in variants(o):
            if v.shape != ta.shape:
                continue
            d = v - ta
            frac = float((np.abs(d - np.median(d)) > 1e-4).mean())
            if best is None or frac < best[1]:
                best = (na, frac, float(np.median(d)))
    if best is None or best[1] > 1e-3:
        print("BAD", n.name, o.shape, best); bad += 1
        if bad >= 4: break
    else:
        ok += 1
print("ok", ok, "bad", bad)
