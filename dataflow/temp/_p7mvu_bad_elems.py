import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
import golden_per_partition as gp
import tap_compare as tc
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import glob

SW = "finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934/intermediate_models/assign_stage_partition_ids_8way.onnx"
PART = glob.glob("finn_deployment_outputs/S12_dense_256_u4_u8in_int6_p7mvu_milpfold_probe7_*/intermediate_models/supported_op_partitions/partition_7.onnx")[0]
taps = Path(sys.argv[1]); case = sys.argv[2]
u8 = np.fromfile(f"/tmp/chain_mixed_all/{case}/in.raw", np.uint8)
ctx = {}
gp.software_boundaries(Path(SW), u8, None, full_ctx=ctx)
pm = ModelWrapper(PART)
nodes = {n.name: n for n in pm.graph.node}
tj = {t["stream"]: t for t in json.loads((taps / "taps.json").read_text())}
nm, swn = "MVAU_rtl_1", "MVAU_84_out0"
nd = nodes[f"GenericPartition_7_{nm}"]
inst = getCustomOp(nd)
dt = inst.get_output_datatype(0)
pe = inst.get_folded_output_shape(0)[-1]
shape = inst.get_normal_output_shape(0)
t = tj[f"GenericPartition_7_{nm}_out_V"] if f"GenericPartition_7_{nm}_out_V" in tj else tj[f"GenericPartition_7_{nm}_out0_V"]
raw = np.fromfile(taps / f"tap_{t['idx']}.bin", np.uint8)
v = tc.decode(raw, t["nb"], pe, dt.bitwidth(), dt.signed()).reshape(shape)
r = np.rint(np.asarray(ctx[swn])).astype(np.int64).reshape(shape)
bad = np.argwhere(v != r)
print(nm, "shape", shape, "n_bad", len(bad))
pix = sorted(set((b[1], b[2]) for b in bad))
print("distinct bad pixels", len(pix), "channels", np.bincount(bad[:, 3], minlength=4).tolist())
lin = [h * 128 + w for h, w in pix]
print("linear pixel idx (first 40):", lin[:40])
print("rows with bad:", sorted(set(h for h, w in pix))[:40])
d = (v - r)[tuple(bad.T)]
print("diff hist:", dict(zip(*np.unique(d, return_counts=True))) if len(d) < 1000 else "many")
for b in bad[:20]:
    print("  idx", b.tolist(), "rtl", int(v[tuple(b)]), "sw", int(r[tuple(b)]))
# input to MVAU_rtl_1 (Thresholding_rtl_3 out) at bad pixels
