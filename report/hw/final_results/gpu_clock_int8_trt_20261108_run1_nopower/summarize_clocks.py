"""Align the NVML clock samples with the benchmark's timed stream windows. usage: summarize_clocks.py <dir with clocks.csv + run.log> [max_sm_mhz]"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

d = Path(sys.argv[1])
MAX_SM = float(sys.argv[2]) if len(sys.argv) > 2 else 2100.0
BITS = {0x1: "GpuIdle", 0x2: "AppClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x10: "SyncBoost", 0x20: "SwThermalSlowdown",
        0x40: "HwThermalSlowdown", 0x80: "HwPowerBrake", 0x100: "DisplayClockSetting"}

rows = []
for r in csv.DictReader(open(d / "clocks.csv")):
    try:
        rows.append({"t": float(r["unix_s"]), "sm": float(r["sm_mhz"]), "mem": float(r["mem_mhz"]), "p": float(r["power_w"]) if r["power_w"] else None,
                     "T": float(r["temp_c"]) if r["temp_c"] else None, "u": float(r["util_pct"]) if r["util_pct"] else None, "thr": int(r["throttle"]) if r["throttle"] else 0})
    except ValueError:
        pass

log = []
for line in open(d / "run.log"):
    t, _, rest = line.rstrip("\n").partition(" ")
    try:
        log.append((float(t), rest))
    except ValueError:
        pass

t_start = next(t for t, s in log if s.startswith("START"))
t_end = next(t for t, s in log if s.startswith("END"))
streams = []
for t, s in log:
    m = re.match(r"stream_warm run (\d+)/(\d+): ([\d.]+) s \(wall ([\d.]+) s\)", s)
    if m:
        streams.append((int(m.group(1)), t - float(m.group(4)), t, float(m.group(3))))
lat = next((float(m.group(1)) for _, s in log for m in [re.search(r"'mode': 'single_image'.*?'latency_ms_mean': '?([\d.]+)", s)] if m), None)


def thr_names(mask):
    return "+".join(n for b, n in BITS.items() if mask & b) or "none"


def stats(label, t0, t1):
    w = [r for r in rows if t0 <= r["t"] <= t1]
    if not w:
        return f"{label:34s} no samples"
    sm = [r["sm"] for r in w]
    pw = [r["p"] for r in w if r["p"] is not None]
    tp = [r["T"] for r in w if r["T"] is not None]
    ut = [r["u"] for r in w if r["u"] is not None]
    thr = 0
    for r in w:
        thr |= r["thr"]
    return (f"{label:34s} n={len(w):4d} {t1 - t0:6.2f}s  SM mean {st.fmean(sm):6.0f} med {st.median(sm):6.0f} min {min(sm):5.0f} max {max(sm):5.0f} MHz "
            f"({100 * st.fmean(sm) / MAX_SM:4.0f}% of {MAX_SM:.0f})  mem {st.fmean(r['mem'] for r in w):5.0f}  P {st.fmean(pw):5.1f} W (max {max(pw):4.1f})  "
            f"T {tp[0]:.0f}->{tp[-1]:.0f} C  util {st.fmean(ut):3.0f}%  throttle: {thr_names(thr)}")


out = [f"GPU clock summary, {len(rows)} NVML samples, median interval {1000 * st.median(b['t'] - a['t'] for a, b in zip(rows, rows[1:])):.0f} ms", ""]
out.append(stats("idle before run (3 s)", t_start - 3.0, t_start))
for k, t0, t1, tot in streams:
    out.append(stats(f"stream_warm {k}/{len(streams)} (cuda {tot:.3f} s)", t0, t1))
if streams:
    out.append(stats("all timed streams", streams[0][1], streams[-1][2]))
    if lat:
        n_single = 210                                           # warmup 10 + single-n 200
        out.append(stats(f"single_image phase (~{n_single * lat / 1000:.2f} s, approx)", streams[-1][2], streams[-1][2] + n_single * lat / 1000))
out.append(stats("whole script (load .. accuracy)", t_start, t_end))
out.append(stats("idle after run (3 s)", t_end, t_end + 3.0))
txt = "\n".join(out)
(d / "summary.txt").write_text(txt + "\n")
print(txt)
