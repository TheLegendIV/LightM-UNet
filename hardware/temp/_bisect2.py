import sys, numpy as np
from collections import defaultdict
exec(open("_bisect.py").read().split("bad = 0")[0])
by_shape = defaultdict(list)
for k, v in ca.items():
    v = np.asarray(v)
    if v.size > 1 and v.dtype.kind == "f":
        by_shape[v.shape].append((k, v))
def match(v, tol):
    v = np.asarray(v)
    best = None
    for k, r in by_shape.get(v.shape, []):
        d = float(np.abs(r - v).max())
        if best is None or d < best[1]:
            best = (k, d)
    return best
shown = 0
for n in b.graph.node:
    if n.op_type != "MultiThreshold":
        continue
    t = n.output[0]
    chain = [t]
    for _ in range(2):
        nxt = [m for m in b.graph.node if t in m.input and m.op_type in ("Add", "Mul")]
        if not nxt:
            break
        t = nxt[0].output[0]; chain.append(t)
    best = min((match(cb[c], 0) for c in chain), key=lambda x: 1e9 if x is None else x[1])
    status = "OK " if best and best[1] < 1e-3 else "BAD"
    print(status, n.name, np.shape(cb[chain[-1]]), "best raw", best)
    if status == "BAD":
        shown += 1
        if shown >= 3:
            break
