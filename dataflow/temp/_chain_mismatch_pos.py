import glob, os
import numpy as np
d = np.load("quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in_verify_ref.npz")
pred = d["pred"]
cnt = np.zeros((256, 256), np.int64)
n = 0
tot_bad = 0
corner = 0
for i in range(165):
    f = f"/tmp/chain_mixed_all/case{i}/GenericPartition_7_out.raw"
    if not os.path.exists(f):
        continue
    r = np.fromfile(f, np.uint8)
    if r.size != 65536:
        continue
    r = r.reshape(256, 256)
    bad = r != pred[i]
    cnt += bad
    n += 1
    tot_bad += int(bad.sum())
    ys, xs = np.nonzero(bad)
    c = int(((ys >= 252) & (xs >= 252)).sum())
    corner += c
    if i < 6 or i in (21, 25, 26, 106):
        print(f"case{i}: bad {int(bad.sum())} corner(>=252,>=252) {c} positions {list(zip(ys.tolist(), xs.tolist()))[:8]} rtl {r[bad][:8].tolist()} torch {pred[i][bad][:8].tolist()}")
print(f"cases {n}: total bad {tot_bad}, in bottom-right 4x4 corner {corner}")
ys, xs = np.nonzero(cnt)
order = np.argsort(-cnt[ys, xs])[:12]
print("top mismatch positions (y,x,count):", [(int(ys[o]), int(xs[o]), int(cnt[ys[o], xs[o]])) for o in order])
border = int(cnt[:2].sum() + cnt[-2:].sum() + cnt[:, :2].sum() + cnt[:, -2:].sum())
print("mismatches within 2px of border:", border, "of", int(cnt.sum()))
