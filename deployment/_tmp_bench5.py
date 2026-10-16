import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"c:\DEV\repos\LightM-UNet\deployment")
import collect_results as cr  # noqa: E402

DS = "Dataset509_ARCADE_1x1_4c"
model = "s12_512_bilinear_int8_maxfps"
images = sorted((cr.NNUNET_RAW / DS / "imagesTs").glob("*.png"))
inputs = cr.preprocess(images, cr.NNUNET_PREPROCESSED / DS / f"img_intf_{model}", model)
frames = [p.read_bytes() for p in inputs]
n = len(frames)

runs = []
for k in range(5):
    r = cr.run_split_benchmark("192.168.0.111", 7, frames, dma_mode=cr.DMA_MODE_BATCH, repeats=10, width=512,
                               height=512, log=lambda s: None)
    first, full = cr.ticks_to_intervals(r["latency_ticks"])
    _, run_full = cr.ticks_to_intervals(r["run_ticks"])
    runs.append({"lat_full": full, "lat_first": first, "run_full": run_full,
                 "outputs_hash": hash(b"".join(r["outputs"]))})
    print(f"run {k + 1}: split ticks per pass {sorted(set(run_full.tolist()))}, "
          f"latency ticks min/max {full.min()}/{full.max()}", flush=True)

clk = cr.PL_CLOCK_MHZ * 1e6
rows = {
    "split_seconds": [r["run_full"].mean() / clk for r in runs],
    "split_fps": [n / (r["run_full"].mean() / clk) for r in runs],
    "hw_infer_ms": [(r["run_full"].mean() - r["lat_full"].mean()) / (n - 1) / clk * 1e3 for r in runs],
    "hw_latency_ms": [r["lat_full"].mean() / clk * 1e3 for r in runs],
    "hw_first_output_ms": [r["lat_first"].mean() / clk * 1e3 for r in runs],
}
print("\nper run:")
for k, v in rows.items():
    print(f"  {k:20s}", "  ".join(f"{x:.6f}" for x in v))
print("\nacross the 5 runs (50 timed passes, 1500 latency samples):")
print(f"  {'metric':20s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s} {'spread %':>9s}")
for k, v in rows.items():
    v = np.array(v)
    print(f"  {k:20s} {v.mean():12.6f} {v.std(ddof=1):12.6f} {v.min():12.6f} {v.max():12.6f} "
          f"{(v.max() - v.min()) / v.mean() * 100:9.4f}")
all_pass = np.concatenate([r["run_full"] for r in runs])
all_lat = np.concatenate([r["lat_full"] for r in runs])
print(f"\nall 50 timed passes: {all_pass.min()} .. {all_pass.max()} ticks (distinct values: {len(set(all_pass.tolist()))})")
print(f"all {len(all_lat)} latency samples: {all_lat.min()} .. {all_lat.max()} ticks (distinct: {len(set(all_lat.tolist()))})")
print("identical class maps in every run:", len({r['outputs_hash'] for r in runs}) == 1)
