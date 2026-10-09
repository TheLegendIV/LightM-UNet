import os, sys, numpy as np
sys.path.insert(0, os.path.abspath("deployment/image_transfer_interface/interfaces"))
from async_read_write_single import BatchImageStreamer

d = "hardware/temp/_probe_in"; o = "hardware/temp/_probe_out"
os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(0)
cases = {"zeros": np.zeros(65536, np.uint8), "ones128": np.full(65536, 128, np.uint8),
         "max255": np.full(65536, 255, np.uint8), "random": rng.integers(0, 256, 65536, dtype=np.uint8)}
for k, v in cases.items():
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    v.tofile(os.path.join(d, f"{k}.raw"))
    BatchImageStreamer("192.168.0.111", 7, d, o, pattern="*.raw").run()
    out = np.fromfile(os.path.join(o, f"{k}.raw"), np.uint8)
    print("RESULT", k, np.bincount(out, minlength=5)[:5], "nonzero idx", np.nonzero(out)[0][:8])
