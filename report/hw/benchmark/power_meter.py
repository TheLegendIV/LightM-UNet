"""Energy per image for the CPU / GPU benchmark rows: active energy minus idle energy.

  gpu  NVML cumulative board-energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ) of the RTX 2050. Board power only, no CPU / DRAM share.
  cpu  Windows "Energy Meter" RAPL_Package0_PKG (cores + uncore + iGPU, no DRAM / wall power). WSL2 containers have no /sys/class/powercap, so report/MILP/benchmark/host_rapl_sampler.ps1 must be
       running on the Windows host; it logs (unix_s, cumulative energy in pWh) to report/MILP/benchmark/power_logs/rapl.csv, which the container reads through the bind mount.

summarize(): idle window (model loaded, nothing running) -> the 5 timed 300-image streams, each bracketed by energy reads -> idle window. Per run, dynamic energy = E_run - P_idle * t_run,
P_idle = mean of the two idle windows; J/image = that / 300, median over the runs (the laptop CPU bursts on the first stream and then throttles, so the mean mixes two regimes; every run is kept in the env JSON). Energy is taken over the same streams that produce the latency row, so both see the same thermal state
(a longer sustained window throttles the laptop CPU and measures a different regime). The GPU side needs no sampler thread, only two counter reads per window."""
from __future__ import annotations

import bisect
import time
from pathlib import Path

RAPL_LOG = Path(__file__).resolve().parent / "power_logs" / "rapl.csv"
PWH_TO_J = 3.6e-9                                  # Energy Meter "Energy" counter unit: picowatt-hours


class GpuMeter:
    name = "nvml-board"

    def __init__(self):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)          # raises if the counter is unsupported
        self.marks = {}                                             # mark time -> cumulative counter in J. No sampler thread: the counter is cumulative, so two reads bound any window

    def start(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def stop(self):
        pass

    def mark(self):
        t = time.time()
        self.marks[t] = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0
        return t

    def energy_j(self, t0, t1):
        return self.marks[t1] - self.marks[t0]

    def info(self):
        return {"energy_source": "NVML nvmlDeviceGetTotalEnergyConsumption (RTX 2050 board energy)"}


class HostRaplMeter:
    name = "rapl-pkg"

    def __init__(self, log=RAPL_LOG):
        self.log = Path(log)

    def _read(self):
        t, e = [], []
        for line in self.log.read_text().splitlines()[1:]:
            p = line.split(",")
            if len(p) >= 2:
                try:
                    t.append(float(p[0])); e.append(float(p[1]) * PWH_TO_J)
                except ValueError:
                    pass                                            # half-written last line
        return t, e

    @property
    def flag(self):
        return self.log.parent / "sample.flag"

    def start(self):
        if not self.log.is_file():
            raise RuntimeError(f"{self.log} missing: start host_rapl_sampler.ps1 on the Windows host first")
        self.resume()

    def resume(self):
        self.flag.touch()
        deadline = time.time() + 5
        while time.time() < deadline:
            t, _ = self._read()
            if t and time.time() - t[-1] < 0.5:
                lag = time.time() - t[-1]
                if lag < -0.5:
                    raise RuntimeError(f"container clock is {-lag:.2f} s behind the host sampler")
                return
            time.sleep(0.1)
        raise RuntimeError("host sampler not producing samples: is host_rapl_sampler.ps1 running (and are the clocks in sync)?")

    def pause(self):
        self.flag.unlink(missing_ok=True)

    def mark(self):
        return time.time()

    def stop(self):
        self.pause()

    def energy_j(self, t0, t1):
        deadline = time.time() + 10
        while True:
            t, e = self._read()
            if t and t[-1] >= t1 + 0.1:
                break
            if time.time() > deadline:
                raise RuntimeError("host sampler stopped before the end of the window")
            time.sleep(0.1)
        return _interp(t, e, t1) - _interp(t, e, t0)

    def info(self):
        return {"energy_source": "Windows Energy Meter RAPL_Package0_PKG (CPU package incl. iGPU, no DRAM / wall)"}


def _interp(ts, es, x):
    i = bisect.bisect_left(ts, x)
    if i == 0 or i >= len(ts):
        raise RuntimeError("energy window outside the sampled range")
    f = (x - ts[i - 1]) / (ts[i] - ts[i - 1])
    return es[i - 1] + f * (es[i] - es[i - 1])


def make_meter(device):
    return GpuMeter() if device == "cuda" else HostRaplMeter()


def idle_window(meter, seconds):
    t0 = meter.mark()
    time.sleep(seconds)
    return t0, meter.mark()


def summarize(meter, windows, idle1, idle2, n):
    """windows = [(t0, t1)] of the timed stream runs (n images each); idle1 / idle2 = idle windows before the first and after the last. Per-run dynamic energy = E_run - P_idle * t_run, P_idle = mean of the
    two idle windows; the J/image value is the median over the runs (the laptop CPU bursts on the first stream, then throttles: the mean mixes two regimes; every run goes to the env JSON)."""
    p_idle = [meter.energy_j(*w) / (w[1] - w[0]) for w in (idle1, idle2)]
    p_idle_mean = sum(p_idle) / 2
    e_run = [meter.energy_j(t0, t1) for t0, t1 in windows]
    t_run = [t1 - t0 for t0, t1 in windows]
    tot = [e / n for e in e_run]
    dyn = [(e - p_idle_mean * t) / n for e, t in zip(e_run, t_run)]
    mean = lambda v: sum(v) / len(v)
    std = lambda v: (sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2
    pw = [e / t for e, t in zip(e_run, t_run)]
    out = {"energy_j_per_image": med(tot), "energy_dyn_j_per_image": med(dyn), "power_active_w": med(pw), "power_idle_w": p_idle_mean, "energy_dyn_j_per_image_mean": mean(dyn),
           "energy_dyn_j_per_image_std": std(dyn), "energy_dyn_j_per_image_runs": dyn, "energy_j_per_image_runs": tot, "power_idle_before_w": p_idle[0], "power_idle_after_w": p_idle[1],
           "idle_window_s": idle1[1] - idle1[0], **meter.info()}
    print(f"energy over the {len(windows)} timed streams: active {out['power_active_w']:.2f} W, idle {p_idle[0]:.2f} W before / {p_idle[1]:.2f} W after; "
          f"{out['energy_j_per_image'] * 1000:.2f} mJ/img total (median), {out['energy_dyn_j_per_image'] * 1000:.2f} mJ/img above idle (median; mean {out['energy_dyn_j_per_image_mean'] * 1000:.2f} +- {out['energy_dyn_j_per_image_std'] * 1000:.2f}) "
          f"(per run: {', '.join(f'{d * 1000:.1f}' for d in dyn)})", flush=True)
    return out
