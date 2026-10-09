import numpy as np
from pathlib import Path

I = Path("deployment/image_transfer_interface")
for n in range(1, 11):
    s = f"test_{n}_p0000_0000"
    g = np.fromfile(I / "output/board_ref" / f"{s}_input_u8.raw", np.uint8).reshape(256, 256)
    r = np.fromfile(I / "reference_256" / f"{s}_ref_pred.raw", np.uint8).reshape(256, 256)
    bad = g != r
    bad[-2:, -2:] = False
    ys, xs = np.nonzero(bad)
    if len(ys):
        print(s, "non-corner mismatches", len(ys), "bbox y", ys.min(), ys.max(), "x", xs.min(), xs.max(),
              "pairs(ref->board)", {f"{int(a)}->{int(b)}": int(c) for (a, b), c in zip(*np.unique(np.stack([r[bad], g[bad]], 1), axis=0, return_counts=True))} if False else
              dict(zip(*[x.tolist() for x in np.unique([f"{a}>{b}" for a, b in zip(r[bad], g[bad])], return_counts=True)])))
    else:
        print(s, "no non-corner mismatches")
