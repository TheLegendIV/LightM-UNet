"""NVML sampler for GPU 0: unix time, SM / memory clock, board power, temperature, utilisation, P-state, throttle-reason bitmask. Runs inside lightmunet_dev next to a GPU benchmark.
usage: gpu_clock_sampler.py out.csv interval_s   (SIGTERM stops)."""
import signal
import sys
import time

import pynvml as nv

out, interval = sys.argv[1], float(sys.argv[2])
nv.nvmlInit()
h = nv.nvmlDeviceGetHandleByIndex(0)
stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

q = [
    lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
    lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
    lambda: round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 3),
    lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
    lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu,
    lambda: nv.nvmlDeviceGetPerformanceState(h),
    lambda: nv.nvmlDeviceGetCurrentClocksThrottleReasons(h),
]


def poll():
    row = []
    for f in q:
        try:
            row.append(f())
        except nv.NVMLError:
            row.append("")
    return row


with open(out, "w") as f:
    f.write("unix_s,sm_mhz,mem_mhz,power_w,temp_c,util_pct,pstate,throttle\n")
    nxt = time.time()
    while not stop:
        t = time.time()
        f.write(",".join(str(v) for v in [f"{t:.4f}"] + poll()) + "\n")
        nxt += interval
        d = nxt - time.time()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.time()
    f.flush()
