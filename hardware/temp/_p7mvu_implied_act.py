import json, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
import tap_compare as tc
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

taps = Path("/tmp/taps_p7mvu")
PART = glob.glob("finn_deployment_outputs/S12_dense_256_u4_u8in_int6_p7mvu_milpfold_probe7_*/intermediate_models/supported_op_partitions/partition_7.onnx")[0]
pm = ModelWrapper(PART)
nodes = {n.name: n for n in pm.graph.node}
tj = {t["stream"]: t for t in json.loads((taps / "taps.json").read_text())}
def get(nm):
    nd = nodes[f"GenericPartition_7_{nm}"]
    inst = getCustomOp(nd)
    dt = inst.get_output_datatype(0)
    pe = inst.get_folded_output_shape(0)[-1]
    shape = inst.get_normal_output_shape(0)
    k = [s for s in tj if s.startswith(f"GenericPartition_7_{nm}_out")][0]
    t = tj[k]
    raw = np.fromfile(taps / f"tap_{t['idx']}.bin", np.uint8)
    return tc.decode(raw, t["nb"], pe, dt.bitwidth(), dt.signed()).reshape(shape)
x = get("Thresholding_rtl_3").reshape(-1)       # 16384 activations
y = get("MVAU_rtl_1").reshape(-1, 4)
nd = nodes["GenericPartition_7_MVAU_rtl_1"]
W = pm.get_initializer(nd.input[1])
print("W shape", None if W is None else W.shape)
w = np.asarray(W).reshape(-1) if W is not None else None
print("w", w)
# the MVAU_rtl_1 input may come through FIFO: assume same order. Expected y = x[:,None]*w[None,:]
exp = x[:, None].astype(np.int64) * w[None, :].astype(np.int64)
print("hw-vs-recomputed-from-tap mismatches per channel:", (y != exp).sum(0).tolist())
bad = np.argwhere(y != exp)
print("n_bad", len(bad))
# implied activation for bad elements
for p, c in bad[:25]:
    imp = y[p, c] / w[c] if w[c] else None
    # search which pixel activation equals implied
    cand = [int(q) for q in range(max(0, p - 6), min(len(x), p + 7)) if x[q] * w[c] == y[p, c]]
    print(f"pix {p} ch {c} x={int(x[p])} w={int(w[c])} rtl={int(y[p,c])} implied_x={imp} matching-neighbour-offsets={[q-p for q in cand]}")
# stats of offset
offs = {}
for p, c in bad:
    for q in range(max(0, p - 8), min(len(x), p + 9)):
        if x[q] * w[c] == y[p, c]:
            offs[q - p] = offs.get(q - p, 0) + 1
print("offset histogram (neighbour whose act explains rtl):", dict(sorted(offs.items())))
print("x[p] stats on bad vs all: bad mean", np.mean([x[p] for p, c in bad]), "all mean", x.mean())
