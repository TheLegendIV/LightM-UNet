"""Write z-scored float32 NHWC .npy inputs (and matching label .npy) for board_run.py.

  python make_board_inputs.py <out_dir> [n_images]

Uses the eval split (images not used for calibration). Runs in the Vitis AI container.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/job")
import model as M  # noqa: E402

out_dir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
os.makedirs(os.path.join(out_dir, "in"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

from PIL import Image  # noqa: E402

for f in M._images("eval")[:n]:
    case = os.path.basename(f)[:-len("_0000.png")]
    np.save(os.path.join(out_dir, "in", case + ".npy"), M._zscore(f)[0].permute(1, 2, 0).numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "labels", case + ".npy"), np.asarray(Image.open(os.path.join(M.LABELS, case + ".png"))).astype(np.uint8))
print("wrote", n, "cases to", out_dir)
