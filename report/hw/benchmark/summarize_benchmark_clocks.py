"""Mean clock rate of every benchmark run of run_benchmarks_with_clocks.sh. usage: summarize_benchmark_clocks.py <out_dir> [gpu_max_sm_mhz]
Reads <out_dir>/<run>/run.log (+ gpu_clock.csv for the GPU runs) and <out_dir>/cpu_clock.csv (Windows host sampler, CPU runs); writes <out_dir>/clock_summary.csv and summary.txt.

Primary figure = the mean clock over the five timed stream_warm windows of the run (the windows that produce throughput_fps; a window is [log stamp - wall time, log stamp]):
  GPU  SM clock (NVML), time-weighted mean of the samples inside the windows.
  CPU  effective MHz = "% Processor Performance"/100 x nominal base clock of the core class (logical CPUs 0-7 = P-core threads 2100 MHz, 8-15 = E-cores 1500 MHz on the i7-1260P), averaged over the
       CPUs weighted by their "% Processor Utility" (= the clock of the cores doing the work); the unweighted mean over all 16 logical CPUs and the P-/E-class means are kept as extra columns.
The whole-run column averages every sample between process start and end (GPU: includes model loading and idle; CPU: only the power-meter windows, i.e. idle before / after + the timed streams)."""
from __future__ import annotations

import csv
import re
import statistics as st
import sys
from pathlib import Path

d = Path(sys.argv[1])
GPU_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 2100.0
CPU_NOMINAL = [2100.0] * 8 + [1500.0] * 8
BITS = {0x1: "GpuIdle", 0x2: "AppClocks", 0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x10: "SyncBoost", 0x20: "SwThermal", 0x40: "HwThermal", 0x80: "HwPowerBrake"}
RUNS = ["gpu_fp32_default", "gpu_fp32_strict", "gpu_fp16_autocast", "gpu_int8_trt", "cpu_fp32", "cpu_int8_ort"]


def read_gpu(path):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append({"t": float(r["unix_s"]), "clk": float(r["sm_mhz"]), "p": float(r["power_w"]) if r["power_w"] else None, "T": float(r["temp_c"]) if r["temp_c"] else None,
                        "u": float(r["util_pct"]) if r["util_pct"] else None, "thr": int(r["throttle"]) if r["throttle"] else 0})
        except (ValueError, KeyError):
            pass
    return out


def read_cpu(path):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            perf = [float(r[f"p{i}"]) for i in range(16)]
            util = [float(r[f"u{i}"]) for i in range(16)]
            t = float(r["unix_s"])
        except (ValueError, KeyError):
            continue
        eff = [CPU_NOMINAL[i] * perf[i] / 100.0 for i in range(16)]
        w = sum(util)
        wmean = lambda idx: (sum(eff[i] * util[i] for i in idx) / sum(util[i] for i in idx)) if sum(util[i] for i in idx) > 0 else st.fmean(eff[i] for i in idx)  # noqa: E731
        out.append({"t": t, "clk": wmean(range(16)) if w > 0 else st.fmean(eff), "all": st.fmean(eff), "pc": wmean(range(8)), "ec": wmean(range(8, 16)), "u": st.fmean(min(x, 100.0) for x in util)})
    return out


def parse_log(path):
    start = end = rc = None
    streams, fps = [], None
    for line in open(path, errors="replace"):
        t, _, rest = line.rstrip("\n").partition(" ")
        try:
            t = float(t)
        except ValueError:
            continue
        if rest.startswith("START"):
            start = t
        elif rest.startswith("END"):
            end = t
            m = re.search(r"rc=(\d+)", rest)
            rc = int(m.group(1)) if m else None
        m = re.match(r"stream_warm run (\d+)/(\d+): ([\d.]+) s \(wall ([\d.]+) s\)", rest)
        if m:
            streams.append((int(m.group(1)), t - float(m.group(4)), t))
        m = re.search(r"'mode': 'stream_warm'.*?'throughput_fps': '?([\d.]+)", rest)
        if m:
            fps = float(m.group(1))
    return start, end, rc, streams, fps


def wmean(vals, ts, cap=0.5):
    """time-weighted mean: each sample counts for the time until the next one (capped, so a gap does not dominate)."""
    if not vals:
        return float("nan")
    w = [min(ts[i + 1] - ts[i], cap) for i in range(len(ts) - 1)] + [ts[-1] - ts[-2] if len(ts) > 1 else 1.0]
    return sum(v * x for v, x in zip(vals, w)) / sum(w)


def window(samples, t0, t1, lead=0.0):
    return [s for s in samples if t0 + lead <= s["t"] <= t1]


def thr_names(mask):
    return "+".join(n for b, n in BITS.items() if mask & b) or "none"


cpu_all = read_cpu(d / "cpu_clock.csv") if (d / "cpu_clock.csv").is_file() else []
cpu_iv = st.median(b["t"] - a["t"] for a, b in zip(cpu_all, cpu_all[1:]) if b["t"] - a["t"] < 2) if len(cpu_all) > 2 else 0.25

rows, text = [], ["Mean clock rate per benchmark run (primary = mean over the five timed stream_warm windows)", ""]
for run in RUNS:
    rd = d / run
    if not (rd / "run.log").is_file():
        continue
    start, end, rc, streams, fps = parse_log(rd / "run.log")
    dev = "cuda" if run.startswith("gpu") else "cpu"
    if dev == "cuda":
        samples, lead = read_gpu(rd / "gpu_clock.csv"), 0.0
    else:
        samples, lead = cpu_all, cpu_iv                              # a CPU sample is the average of the interval that ENDS at its stamp
    win = [window(samples, t0, t1, lead) for _, t0, t1 in streams]
    allw = [s for w in win for s in w]
    whole = window(samples, start, end)
    row = {"run": run, "device": dev, "rc": rc, "throughput_fps": fps, "timed_streams": len(streams), "timed_s": round(sum(t1 - t0 for _, t0, t1 in streams), 2), "n_samples_timed": len(allw)}
    if allw:
        ts = [s["t"] for s in allw]
        clk = [s["clk"] for s in allw]
        row["clock_mean_mhz"] = round(wmean(clk, ts), 1)
        row["clock_median_mhz"] = round(st.median(clk), 1)
        row["clock_min_mhz"], row["clock_max_mhz"] = round(min(clk), 1), round(max(clk), 1)
        row["clock_std_mhz"] = round(st.pstdev(clk), 1)
        if dev == "cuda":
            row["clock_pct_of_max"] = round(100 * row["clock_mean_mhz"] / GPU_MAX, 1)
            pw = [s["p"] for s in allw if s["p"] is not None]
            row["board_power_w"] = round(wmean(pw, [s["t"] for s in allw if s["p"] is not None]), 1) if pw else ""
            tp = [s["T"] for s in allw if s["T"] is not None]
            row["temp_c_start"], row["temp_c_end"] = (tp[0], tp[-1]) if tp else ("", "")
            ut = [s["u"] for s in allw if s["u"] is not None]
            row["util_pct"] = round(st.fmean(ut), 0) if ut else ""
            thr = 0
            for s in allw:
                thr |= s["thr"]
            row["throttle_reasons"] = thr_names(thr)
        else:
            row["clock_mean_all_cores_mhz"] = round(wmean([s["all"] for s in allw], ts), 1)
            row["p_core_mhz"] = round(wmean([s["pc"] for s in allw], ts), 1)
            row["e_core_mhz"] = round(wmean([s["ec"] for s in allw], ts), 1)
            row["util_pct"] = round(st.fmean(s["u"] for s in allw), 0)
    if whole:
        row["clock_whole_run_mean_mhz"] = round(wmean([s["clk"] for s in whole], [s["t"] for s in whole]), 1)
        row["whole_run_samples"] = len(whole)
    rows.append(row)

    text.append(f"{run}  (rc={rc}, {fps} fps, {len(streams)} timed streams, {row['timed_s']} s, {len(allw)} samples)")
    for (k, t0, t1), w in zip(streams, win):
        if w:
            c = [s["clk"] for s in w]
            text.append(f"   stream {k}: {t1 - t0:6.2f} s  n={len(w):3d}  mean {wmean(c, [s['t'] for s in w]):7.1f}  min {min(c):7.1f}  max {max(c):7.1f} MHz")
        else:
            text.append(f"   stream {k}: {t1 - t0:6.2f} s  no samples")
    text.append("   " + ", ".join(f"{k} {v}" for k, v in row.items() if k.startswith(("clock_", "p_core", "e_core", "board", "util", "temp", "throttle"))))
    text.append("")

fields = ["run", "device", "rc", "throughput_fps", "timed_streams", "timed_s", "n_samples_timed", "clock_mean_mhz", "clock_median_mhz", "clock_min_mhz", "clock_max_mhz", "clock_std_mhz", "clock_pct_of_max",
          "clock_mean_all_cores_mhz", "p_core_mhz", "e_core_mhz", "util_pct", "board_power_w", "temp_c_start", "temp_c_end", "throttle_reasons", "clock_whole_run_mean_mhz", "whole_run_samples"]
with open(d / "clock_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
(d / "summary.txt").write_text("\n".join(text) + "\n")
print("\n".join(text))
