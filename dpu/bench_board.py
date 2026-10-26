"""On-board benchmark of a compiled xmodel on the ZCU104 DPUs (run by dpu/collect_results.py over SSH).

  python3 bench_board.py --model m.xmodel --images test_u8.npy --out bench_out

The xmodel produces the class map on the DPU (argmax appended, see builds/*/add_argmax.py), so the timed unit is
DPU input -> DPU output with the class map already in memory. Preprocessing is NOT timed:
all N images are z-scored and quantized with the DPU input scale before the first timed run and stay resident.

  stream_warm   --repeats timed passes over the whole stream (batch 1, --threads runners, one per DPU core), after
                --warmup untimed inferences; every pass is one region from the first submit to the last result
  single_image  --single-n synchronous runs of one image on one runner (after --warmup), per-run latency
  accuracy      separate untimed pass over all images, class maps -> out/pred.npy (N,H,W) uint8
  power         INA226 on the 12 V board input (hwmon), sampled by a separate process; idle windows (model loaded,
                nothing running) before and after the streams. Samples and window times are written to out/result.json,
                dynamic energy is computed on the host with analysis/benchmark/power_meter.summarize.
"""
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HWMON_NAME = "ina226"
IIO = "/sys/bus/iio/devices/iio:device1"


def find_power_file():
    base = "/sys/class/hwmon"
    for d in sorted(os.listdir(base)):
        if open(f"{base}/{d}/name").read().strip() == HWMON_NAME:
            return f"{base}/{d}/power1_input"             # microwatts
    raise RuntimeError("no ina226 hwmon device")


def sample_power(path):
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    f = open(find_power_file())
    t, w = [], []
    while not stop:
        f.seek(0)
        w.append(int(f.read()) * 1e-6)
        t.append(time.time())
        time.sleep(0.004)
    json.dump({"t": t, "w": w}, open(path, "w"))


def dpu_subgraph(graph):
    subs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [s for s in subs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    assert len(dpu) == 1, f"expected 1 DPU subgraph, got {len(dpu)}"
    return dpu[0]


class Worker:
    def __init__(self, subgraph):
        import vart
        self.runner = vart.Runner.create_runner(subgraph, "run")
        out_t = self.runner.get_output_tensors()[0]
        self.out = [np.empty(tuple(out_t.dims), dtype=np.int8, order="C")]

    def run(self, x):
        self.runner.wait(self.runner.execute_async([x], self.out))

    def class_map(self):
        o = self.out[0][0]
        return o[..., 0].astype(np.uint8) if o.shape[-1] == 1 else np.argmax(o, axis=-1).astype(np.uint8)


def prepare(images, in_scale):
    xs = []
    for img in images:
        x = img.astype(np.float32)
        x = (x - x.mean()) / max(float(x.std()), 1e-8)
        q = np.clip(np.round(x * in_scale), -128, 127).astype(np.int8)
        xs.append(np.ascontiguousarray(q[None, :, :, None]))
    return xs


def stream(workers, xs, n):
    """One timed region over n images split across the workers (static strided partition, threads parked on an event)."""
    go, ready = threading.Event(), threading.Barrier(len(workers) + 1)

    def loop(k, w):
        ready.wait()
        go.wait()
        for i in range(k, n, len(workers)):
            w.run(xs[i % len(xs)])

    ts = [threading.Thread(target=loop, args=(k, w)) for k, w in enumerate(workers)]
    [t.start() for t in ts]
    ready.wait()
    w0 = time.time()
    t0 = time.perf_counter()
    go.set()
    [t.join() for t in ts]
    t1 = time.perf_counter()
    return {"total_s": t1 - t0, "t0": w0, "t1": w0 + (t1 - t0)}


def read_iio(name):
    try:
        return open(f"{IIO}/{name}").read().strip()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--images", help="(N,H,W) uint8 .npy")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--single-n", type=int, default=200)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--sample-power", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.sample_power:
        return sample_power(a.sample_power)
    if not (a.model and a.images):
        ap.error("--model and --images are required")

    import xir
    os.makedirs(a.out, exist_ok=True)
    subgraph = dpu_subgraph(xir.Graph.deserialize(a.model))
    workers = [Worker(subgraph) for _ in range(a.threads)]
    in_t = workers[0].runner.get_input_tensors()[0]
    in_scale = 2 ** in_t.get_attr("fix_point")
    images = np.load(a.images)
    n = len(images)
    t = time.time()
    xs = prepare(images, in_scale)
    print(f"{n} images preprocessed + quantized (scale {in_scale}) in {time.time() - t:.1f} s (not timed)", flush=True)

    sampler = None
    idle1 = idle2 = None
    if not a.no_power:
        sampler = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--sample-power", os.path.join(a.out, "power.json")])
        time.sleep(5)                                           # let load activity settle, then idle with the model resident
        i0 = time.time()
        time.sleep(a.idle_s)
        idle1 = (i0, time.time())

    stream(workers, xs, a.warmup)
    runs = []
    for k in range(a.repeats):
        runs.append(stream(workers, xs, n))
        print(f"stream_warm run {k + 1}/{a.repeats}: {runs[-1]['total_s']:.3f} s ({n / runs[-1]['total_s']:.2f} fps)", flush=True)

    if not a.no_power:
        time.sleep(2)                                           # cool-down of the load transient is excluded from the idle window
        i0 = time.time()
        time.sleep(a.idle_s)
        idle2 = (i0, time.time())
        time.sleep(0.2)
        sampler.send_signal(signal.SIGTERM)
        sampler.wait(timeout=30)

    w = workers[0]
    for k in range(a.warmup):
        w.run(xs[0])
    lat = []
    for _ in range(a.single_n):
        t0 = time.perf_counter()
        w.run(xs[0])
        lat.append((time.perf_counter() - t0) * 1000.0)

    pred = np.empty(images.shape, dtype=np.uint8)
    for i in range(n):
        w.run(xs[i])
        pred[i] = w.class_map()
    np.save(os.path.join(a.out, "pred.npy"), pred)

    result = {
        "n_images": n, "threads": a.threads, "warmup": a.warmup, "input_scale": in_scale,
        "runs": runs, "latency_ms": lat, "idle1": idle1, "idle2": idle2,
        "power_file": None if a.no_power else "power.json",
        "info": {"kernel": platform.release(), "python": platform.python_version(),
                 "model": os.path.basename(a.model),
                 "power_source": "INA226 12 V board input (hwmon power1_input), whole-board incl. regulator losses, PS, PL, DDR, fan, PHY",
                 "ina226_sampling_frequency_hz": read_iio("in_sampling_frequency"),
                 "ina226_oversampling_ratio": read_iio("in_oversampling_ratio"),
                 "ina226_integration_time_s": read_iio("in_voltage0_integration_time"),
                 "ina226_shunt_ohm": read_iio("in_shunt_resistor")},
    }
    json.dump(result, open(os.path.join(a.out, "result.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
