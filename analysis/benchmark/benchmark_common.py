"""Shared pieces of the CPU / GPU benchmark tables (copied from the FP32 benchmark script, now in analysis/archive/benchmark/): paths, the CSV schema and its upsert, small helpers.
Rows of analysis/report/MILP/final_results/benchmark_s12_512.csv are keyed by (platform, numeric_mode, mode); the env JSON next to it records software / hardware per device and mode."""
from __future__ import annotations

import csv
import json
import platform as _platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deployment"))
sys.path.insert(0, str(REPO / "analysis" / "501_ARCADE"))
sys.path.insert(0, str(REPO / "analysis" / "archive" / "501_ARCADE"))     # segmentation_topology.py now lives in the archive folder

DATASET = "Dataset509_ARCADE_1x1_4c"
OUT_DIR = REPO / "analysis" / "report" / "MILP" / "final_results"
OUT_CSV = OUT_DIR / "benchmark_s12_512.csv"
OUT_ENV = OUT_DIR / "benchmark_s12_512_env.json"
CLASSES = ["LAD", "RCA", "LCX", "LM"]
COLUMNS = (["platform", "device", "precision", "numeric_mode", "model", "mode", "batch", "n_images", "threads", "runs",
            "latency_ms_mean", "latency_ms_std", "latency_ms_median", "latency_ms_min", "latency_ms_p95", "latency_ms_p99", "throughput_fps", "throughput_fps_std", "total_s", "total_s_std",
            "dice", "dice_binary"] + [f"dice_{c}" for c in CLASSES] + ["timestamp"]
           + ["energy_dyn_j_per_image", "energy_j_per_image", "power_active_w", "power_idle_w"])     # stream_warm rows only; dyn = above idle (see power_meter.py)


def test_images(limit):
    files = sorted((REPO / "data" / "nnUNet_raw" / DATASET / "imagesTs").glob("*.png"))
    return files[:limit] if limit else files


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))]


def fmt(x, nd=4):
    return "" if x is None else round(float(x), nd)


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return _platform.processor() or "unknown CPU"


def upsert(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(csv.DictReader(open(OUT_CSV, newline=""))) if OUT_CSV.is_file() else []
    keys = {(r["platform"], r["numeric_mode"], r["mode"]) for r in rows}
    kept = [r for r in existing if (r["platform"], r["numeric_mode"], r["mode"]) not in keys]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(kept + [{c: ("" if r.get(c) is None else r[c]) for c in COLUMNS} for r in rows])
