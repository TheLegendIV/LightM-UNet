"""Append the FPGA board rows from deployment/results.csv to the CPU/GPU benchmark table (analysis/report/MILP/final_results/benchmark_s12_512.csv), same schema and mode names.

Source row: one (model_name, config) of deployment/results.csv, default s12_512_bilinear_int8_maxfps / board_ipool. All its timings are PL-clock ticks of the axi_late_mon block (no CPU timer, no network):
  stream_warm   <- split_seconds (+std), split_fps: first input beat -> last output beat of the 300-image split, mean over split_repeats passes. Inputs are already in board DDR, the host
                   transfer is outside the window, the argmax is inside the chain, and every pass includes the pipeline fill (about one latency), as in the CPU/GPU rows.
  single_image  <- hw_latency_ms: mean single-image latency (input first beat -> output last beat) over the latency pass; (the max, hw_latency_max_ms, is not a column here).
The steady-state figure hw_infer_ms/hw_infer_fps (split time minus one latency, per image) is not a column of this table; it is printed for reference.

Run inside the lightmunet_dev container (reuses the table writer of benchmark_fp32_cpu_gpu.py):
    python analysis/benchmark/add_fpga_rows.py [--model-name s12_512_bilinear_int8_maxfps] [--config board_ipool]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_fp32_cpu_gpu as bench  # noqa: E402

RESULTS = bench.REPO / "deployment" / "results.csv"


def f(row: dict, key: str):
    v = row.get(key, "")
    return None if v in ("", None) else float(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="s12_512_bilinear_int8_maxfps")
    ap.add_argument("--config", default="board_ipool")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(RESULTS, newline="")) if (r["model_name"], r["config"]) == (a.model_name, a.config)]
    if not rows:
        sys.exit(f"({a.model_name}, {a.config}) not in {RESULTS}")
    r = rows[0]
    n = int(r["n_scored"])
    if f(r, "split_seconds") is None or f(r, "hw_latency_ms") is None:
        sys.exit("that row has no split/latency timings")
    secs, secs_std = f(r, "split_seconds"), f(r, "split_seconds_std") or 0.0
    acc = {k: bench.fmt(f(r, k)) for k in ["dice", "dice_binary"] + [f"dice_{c}" for c in bench.CLASSES]}
    base = dict(platform="Zynq UltraScale+ ZCU7EV (FPGA)", device="fpga", precision="INT8 (W8A8)", numeric_mode="int8", model=f"{a.model_name} ({a.config})", batch="", n_images=n,
                threads="", timestamp=r["timestamp"], **acc)
    out = [
        {**base, "mode": "stream_warm", "runs": int(float(r["split_repeats"])), "total_s": bench.fmt(secs), "total_s_std": bench.fmt(secs_std, 5), "latency_ms_mean": bench.fmt(secs / n * 1000.0),
         "latency_ms_std": bench.fmt(secs_std / n * 1000.0), "throughput_fps": bench.fmt(f(r, "split_fps"), 3)},
        {**base, "mode": "single_image", "runs": n, "latency_ms_mean": bench.fmt(f(r, "hw_latency_ms"))},
    ]
    bench.upsert(out)
    print(f"added {len(out)} FPGA rows from deployment/results.csv ({a.model_name}, {a.config}); steady state without pipeline fill: "
          f"{r.get('hw_infer_ms')} ms/image = {r.get('hw_infer_fps')} fps; first output {r.get('hw_first_output_ms')} ms; max latency {r.get('hw_latency_max_ms')} ms")


if __name__ == "__main__":
    main()
