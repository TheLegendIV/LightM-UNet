"""INT8 CPU / GPU benchmark of the uniform-INT8 QAT network (checkpoint nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_12_dense_relu_warmstart150ep_bilinear_uniform_int8_
perlayer_ft15ep, the network of the FPGA build s12_512_bilinear_int8_maxfps) -> rows of analysis/report/MILP/final_results/benchmark_s12_512.csv, same modes and timing rules as benchmark_fp32_cpu_gpu.py.

  --backend ort-cpu   ONNX Runtime CPU, int8 kernels (QDQ graph -> QLinearConv, VNNI), i7-1260P
  --backend trt-gpu   TensorRT INT8 (explicit quantization; no FP16 flag, so layers without an int8 kernel run FP32), RTX 2050 laptop (TensorRT only exists for NVIDIA GPUs)
The ONNX files are analysis/benchmark/export_int8_qcdq.py's QDQ conversion of the FPGA build's QONNX export (BatchNorm folded into per-channel int8 weight scales, an ArgMax appended: the timed unit is
network + argmax, as everywhere else). The input is the board's UINT8 input contract (z-score, quantize with the input_profiles.json scale, +128) computed on the host BEFORE timing, as the FPGA host does.

Timing (as the FP32 benchmark and the FPGA window): all 300 inputs resident on the device, ONE timed region from the first to the last inference (stream_warm, after --warmup untimed runs), batch 1,
no host-to-device or result transfer inside it; single_image = one image, synchronized per run (mean/std/median/min/p95/p99). Accuracy: separate untimed pass, same Dice definitions as deployment/collect_results.py.

Usage (lightmunet_dev; TensorRT is installed to /opt/trt_pkgs, see TRT_PKGS):
    python analysis/benchmark/benchmark_int8_cpu_gpu.py --backend ort-cpu
    PYTHONPATH=/opt/trt_pkgs python analysis/benchmark/benchmark_int8_cpu_gpu.py --backend trt-gpu
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_common as bench  # noqa: E402  (table writer, helpers)
from img_intf.utils.preprocess import png_to_u8  # noqa: E402

PROFILE = "s12_512_bilinear_int8_maxfps"
CHECKPOINT = ("nnUNetTrainerLayerQuantEnetFINN_12_dense_relu_bilinear_upsample_256_perlayer_12_dense_relu_warmstart150ep_bilinear_uniform_int8_perlayer_ft15ep")
ONNX_DIR = HERE / "int8"


def load_inputs(limit):
    files = bench.test_images(limit)
    x = np.stack([png_to_u8(p, PROFILE).astype(np.float32) for p in files])[:, None]      # (N,1,512,512) uint8 levels as float32 (the model's input type)
    return files, x


# ------------------------------------------------------------------------------------------------ ONNX Runtime CPU
class OrtCpu:
    name = "ort-cpu"

    def __init__(self, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.ort = ort
        self.sess = ort.InferenceSession(str(ONNX_DIR / "s12_512_bilinear_int8_qcdq_ort.onnx"), so, providers=["CPUExecutionProvider"])
        self.threads = threads or so.intra_op_num_threads or 0
        self.in_name, self.out_name = self.sess.get_inputs()[0].name, self.sess.get_outputs()[0].name
        self.out = ort.OrtValue.ortvalue_from_shape_and_type([1, 512, 512], np.int64, "cpu", 0)

    def prepare(self, x):
        self.inputs = [self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x[i:i + 1])) for i in range(x.shape[0])]
        self.io = self.sess.io_binding()
        self.io.bind_output(self.out_name, "cpu", 0, np.int64, [1, 512, 512], self.out.data_ptr())

    def run_one(self, i):
        self.io.bind_ortvalue_input(self.in_name, self.inputs[i])
        self.sess.run_with_iobinding(self.io)

    def sync(self):
        pass

    def predict(self, i):
        self.run_one(i)
        return self.out.numpy()[0].astype(np.uint8).copy()

    def stream_run(self, n):
        t0 = time.perf_counter()
        for i in range(n):
            self.run_one(i)
        t = time.perf_counter() - t0
        return {"total_s": t, "wall_s": t}

    def info(self):
        return {"runtime": f"onnxruntime {self.ort.__version__}", "threads": self.threads, "providers": self.sess.get_providers()}


# ------------------------------------------------------------------------------------------------ TensorRT GPU
class TrtGpu:
    name = "trt-gpu"

    def __init__(self, threads):
        import tensorrt as trt
        import torch
        self.trt, self.torch = trt, torch
        logger = trt.Logger(trt.Logger.WARNING)
        plan = ONNX_DIR / "s12_512_bilinear_int8only_trt.plan"
        runtime = trt.Runtime(logger)
        if plan.is_file():
            engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        else:
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse((ONNX_DIR / "s12_512_bilinear_int8_qcdq_trt.onnx").read_bytes()):
                raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.INT8)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED     # per-layer precision readable with the engine inspector
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            ser = builder.build_serialized_network(network, config)
            if ser is None:
                raise RuntimeError("TensorRT engine build failed")
            plan.write_bytes(bytes(ser))
            engine = runtime.deserialize_cuda_engine(bytes(ser))
        self.engine = engine
        self.ctx = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        out_dtype = {trt.int32: torch.int32, trt.int64: torch.int64, trt.int8: torch.int8, trt.uint8: torch.uint8}[engine.get_tensor_dtype(self.out_name)]
        self.out = torch.empty(tuple(engine.get_tensor_shape(self.out_name)), dtype=out_dtype, device="cuda")
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        self.stream = torch.cuda.Stream()

    def prepare(self, x):
        self.inputs = [self.torch.from_numpy(np.ascontiguousarray(x[i:i + 1])).cuda() for i in range(x.shape[0])]
        self.torch.cuda.synchronize()

    def run_one(self, i):
        self.ctx.set_tensor_address(self.in_name, self.inputs[i].data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)

    def sync(self):
        self.stream.synchronize()

    def predict(self, i):
        self.run_one(i)
        self.sync()
        return self.out[0].to("cpu").numpy().astype(np.uint8)

    def stream_run(self, n):
        self.sync()
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        w0 = time.perf_counter()
        start.record(self.stream)
        for i in range(n):
            self.run_one(i)
        end.record(self.stream)
        end.synchronize()
        return {"total_s": start.elapsed_time(end) / 1000.0, "wall_s": time.perf_counter() - w0}

    def info(self):
        return {"runtime": f"TensorRT {self.trt.__version__}", "builder_flags": "INT8", "engine_layers": self.engine.num_layers}


def single_latencies(be, warmup, n):
    lat = []
    for k in range(warmup + n):
        be.sync()
        t0 = time.perf_counter()
        be.run_one(0)
        be.sync()
        if k >= warmup:
            lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def accuracy(be, files):
    import segmentation_topology as topo
    from img_intf.utils.preprocess import case_id
    labels_dir = bench.REPO / "data" / "nnUNet_raw" / bench.DATASET / "labelsTs"
    per_class = {c: [] for c in bench.CLASSES}
    binary = []
    for i, p in enumerate(files):
        pred = be.predict(i)
        gt = topo.load_class_id_mask(labels_dir / f"{case_id(p)}.png", binarize=False)
        pred = topo.resize_mask_to(pred, gt.shape)
        for cid, name in enumerate(bench.CLASSES, start=1):
            per_class[name].append(topo.dice_score(gt == cid, pred == cid))
        binary.append(topo.dice_score(gt > 0, pred > 0))
    out = {f"dice_{c}": float(np.mean(per_class[c])) for c in bench.CLASSES}
    out["dice"] = float(np.mean([out[f"dice_{c}"] for c in bench.CLASSES]))
    out["dice_binary"] = float(np.mean(binary))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["ort-cpu", "trt-gpu"], required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--single-n", type=int, default=200)
    ap.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads (default: ORT's own default)")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--power", action="store_true", help="also measure energy per image above idle (power_meter.py; cpu needs host_rapl_sampler.ps1 running on the Windows host)")
    ap.add_argument("--idle-s", type=float, default=20.0)
    a = ap.parse_args()

    files, x = load_inputs(a.limit)
    n = len(files)
    device = "cpu" if a.backend == "ort-cpu" else "cuda"
    be = (OrtCpu if a.backend == "ort-cpu" else TrtGpu)(a.threads)
    be.prepare(x)
    meter = idle1 = None
    if a.power:
        import power_meter as pm
        meter = pm.make_meter(device)
        meter.start()
        time.sleep(5)                                               # let load / allocation activity settle, then record idle with the model resident
        idle1 = pm.idle_window(meter, a.idle_s)
    for k in range(a.warmup):
        be.run_one(k % n)
    be.sync()
    runs, windows = [], []
    for k in range(a.repeats):
        if a.power:
            windows.append([meter.mark()])
        runs.append(be.stream_run(n))
        if a.power:
            windows[-1].append(meter.mark())                        # energy is taken over exactly the timed streams (same thermal state as the latency row)
        print(f"stream_warm run {k + 1}/{a.repeats}: {runs[-1]['total_s']:.3f} s (wall {runs[-1]['wall_s']:.3f} s)", flush=True)
    energy = {}
    if a.power:
        energy = pm.summarize(meter, windows, idle1, pm.idle_window(meter, a.idle_s), n)
        meter.stop()
    lat = single_latencies(be, a.warmup, a.single_n)
    acc = {} if a.skip_accuracy else accuracy(be, files)
    if acc:
        print(f"accuracy ({n} images): dice {acc['dice']:.4f}  binary {acc['dice_binary']:.4f}  " + "  ".join(f"{c} {acc[f'dice_{c}']:.4f}" for c in bench.CLASSES), flush=True)

    plat = bench.cpu_name() if device == "cpu" else __import__("torch").cuda.get_device_name(0)
    mode_id = "int8_ort" if device == "cpu" else "int8_trt"
    info = be.info()
    base = dict(platform=plat, device=device, precision=("INT8 (QAT, ONNX Runtime)" if device == "cpu" else "INT8 (QAT, TensorRT)"), numeric_mode=mode_id, model=CHECKPOINT, batch=1, n_images=n,
                threads=info.get("threads", "") if device == "cpu" else "", timestamp=datetime.now().isoformat(timespec="seconds"), **{k: bench.fmt(v) for k, v in acc.items()})
    ts = [r["total_s"] for r in runs]
    t, sd = statistics.fmean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)
    fps = [n / v for v in ts]
    rows = [
        {**base, "mode": "stream_warm", "runs": len(ts), "total_s": bench.fmt(t), "total_s_std": bench.fmt(sd), "latency_ms_mean": bench.fmt(t / n * 1000.0), "latency_ms_std": bench.fmt(sd / n * 1000.0),
         "throughput_fps": bench.fmt(n / t, 3), "throughput_fps_std": bench.fmt(statistics.stdev(fps) if len(fps) > 1 else 0.0, 3),
         **{k: bench.fmt(energy[k]) for k in ("energy_dyn_j_per_image", "energy_j_per_image", "power_active_w", "power_idle_w") if k in energy}},
        {**base, "mode": "single_image", "runs": len(lat), "latency_ms_mean": bench.fmt(statistics.fmean(lat)), "latency_ms_std": bench.fmt(statistics.stdev(lat)), "latency_ms_median": bench.fmt(statistics.median(lat)),
         "latency_ms_min": bench.fmt(min(lat)), "latency_ms_p95": bench.fmt(bench.pct(lat, 95)), "latency_ms_p99": bench.fmt(bench.pct(lat, 99))},
    ]
    bench.upsert(rows)
    env = bench.json.loads(bench.OUT_ENV.read_text()) if bench.OUT_ENV.is_file() else {}
    env[f"{device}:{mode_id}"] = {"platform": plat, "checkpoint": CHECKPOINT, "input": f"uint8 levels as float32, profile {PROFILE}", "warm_runs_s": ts, "warmup": a.warmup, "batch_size": 1,
                                 "timestamp": datetime.now().isoformat(timespec="seconds"), **info,
                                 **({"energy": {k: v for k, v in energy.items() if k not in ("energy_dyn_j_per_image", "energy_j_per_image", "power_active_w", "power_idle_w")}} if energy else {})}
    bench.OUT_ENV.write_text(bench.json.dumps(env, indent=2))
    for r in rows:
        print({k: r[k] for k in ("mode", "runs", "total_s", "total_s_std", "latency_ms_mean", "latency_ms_std", "latency_ms_median", "throughput_fps") if r.get(k) not in (None, "")}, flush=True)
    print(f"wrote {bench.OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
