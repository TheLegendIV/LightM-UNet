"""Send one reference u8 input to the board server and compare the returned class map with the PyTorch reference.

Run: .venv\\Scripts\\python.exe hardware/temp/_board_single_image_test.py [test_1_p0000_0000]
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IFACE = REPO_ROOT / "deployment" / "image_transfer_interface"
sys.path.insert(0, str(IFACE / "interfaces"))
from async_read_write_single import BatchImageStreamer  # noqa: E402

stem = sys.argv[1] if len(sys.argv) > 1 else "test_1_p0000_0000"
ref_dir = IFACE / "reference_256"
out_dir = IFACE / "output" / "board_single"

BatchImageStreamer("192.168.0.111", 7, str(ref_dir), str(out_dir), pattern=f"{stem}_input_u8.raw").run()

got = np.fromfile(out_dir / f"{stem}_input_u8.raw", dtype=np.uint8)
ref = np.fromfile(ref_dir / f"{stem}_ref_pred.raw", dtype=np.uint8)
cnt = lambda a: {int(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}  # noqa: E731
fg_dice = 2 * float(((got > 0) & (ref > 0)).sum()) / max(float((got > 0).sum() + (ref > 0).sum()), 1.0)
print(f"board classes {cnt(got)}")
print(f"ref   classes {cnt(ref)}")
print(f"pixel agreement {float((got == ref).mean()):.4f} | fg Dice vs ref {fg_dice:.4f}")
