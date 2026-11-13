"""CPU / GPU latency + throughput + accuracy benchmark of the S12 512 bilinear FP32 network -> rows of report/hw/final_results/benchmark_s12_512.csv (the INT8 rows come from
benchmark_int8_cpu_gpu.py, same CSV schema, same timing rules). Ported from the archived analysis/benchmark/benchmark_fp32_cpu_gpu.py: the measurement itself is unchanged; paths follow the current
repo layout, the CSV schema / upsert is benchmark_common's (so the energy columns survive), and --power / --cooldown-s / --idle-s work as in the INT8 script.

Network: nnUNetTrainerENet_12_dense_relu_warmstart150ep (ILP/configs/config_12_dense_relu_warmstart150ep.py), checkpoint_best.pth, 300-image Dataset509 test split (imagesTs).
RAW inference only: the nnU-Net pipeline (file I/O, sliding window, resampling, file export) is not timed; the per-pixel ARGMAX is timed (the FPGA chain includes it). Inputs are z-scored with deployment/img_intf/utils/preprocess.py (the board's
preprocessing) before timing and are already resident on the device (GPU: uploaded once), so host-to-device and result transfers are NOT inside any timed region -- the analogue of the FPGA's
IODMA start -> stop window.

Modes (per device / numeric mode)
  stream_warm   300 images, batch 1, ONE timed region first forward -> last forward (forward + argmax), after --warmup untimed forwards (cudnn.benchmark on): steady state. Model load,
                move to device and the input upload are before the clock. Repeated --repeats times in this process; the MEAN (and std) over the repeats is reported.
  single_image  one fixed image, --warmup then --single-n timed forwards, each synchronized: latency distribution (mean/std/median/min/p95/p99).
  The timed unit everywhere is forward + argmax (infer()). The accuracy pass (same unit, then scoring, same definitions as deployment/collect_results.py) is separate and never timed.

GPU numeric modes (--gpu-mode): fp32_default = PyTorch defaults (cuDNN TF32 convolutions ON, what the project and a stock deployment on Ampere use), fp32_strict = both TF32 flags off,
fp16_autocast = torch.autocast fp16 (what nnU-Net's own predict/trainer use). The CPU is plain FP32 (no TF32).

Usage (inside the lightmunet_dev container, repo at /workspace/LightM-UNet):
    python report/hw/benchmark/benchmark_fp32_cpu_gpu.py --device cuda
    python report/hw/benchmark/benchmark_fp32_cpu_gpu.py --device cuda --gpu-mode fp32_strict
    python report/hw/benchmark/benchmark_fp32_cpu_gpu.py --device cpu --threads 8 --power      # cpu energy needs host_rapl_sampler.ps1 running on the Windows host
    python report/hw/benchmark/benchmark_fp32_cpu_gpu.py --device cuda --limit 10 --repeats 1     # smoke test (still upserts its rows)

Run discipline: laptop plugged in, nothing else using the GPU/CPU; both laptop parts can thermally throttle, so look at the spread of the repeats.
Output: report/hw/final_results/benchmark_s12_512.csv (upserted by (platform, numeric_mode, mode)) and benchmark_s12_512_env.json (software/hardware/flags of each device+mode).
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_common as bench  # noqa: E402  (paths, CSV schema + upsert, helpers; also puts deployment/ and nas/ on sys.path)

REPO = bench.REPO
sys.path.insert(0, str(REPO / "enet"))
sys.path.insert(0, str(REPO / "ILP"))
from nnunetv2.nets.ENet import ENet  # noqa: E402
from img_intf.utils.preprocess import case_id, zscore  # noqa: E402  (the board's own z-score)
import segmentation_topology as topo  # noqa: E402

CONFIG = "config_12_dense_relu_warmstart150ep"
DATASET = bench.DATASET
CHECKPOINT = REPO / "data" / "nnUNet_results" / DATASET / "nnUNetTrainerENet_12_dense_relu_warmstart150ep__nnUNetPlans__2d" / "fold_0" / "checkpoint_best.pth"
CLASSES = bench.CLASSES
PRECISION_LABEL = {
    "fp32": "FP32",
    "fp32_default": "FP32 (PyTorch default, cuDNN TF32 convs)",
    "fp32_strict": "FP32 (strict, TF32 off)",
    "fp16_autocast": "FP16 (torch.autocast)",
}
ENERGY_ROW_KEYS = ("energy_dyn_j_per_image", "energy_j_per_image", "power_active_w", "power_idle_w")


# ------------------------------------------------------------------------------------------------ model / data
def build_model() -> ENet:
    cfg = importlib.import_module(f"configs.{CONFIG}")
    g = lambda k, d=None: getattr(cfg, k, d)  # noqa: E731
    model = ENet(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS, channels=cfg.CHANNELS, bottlenecks_per_stage=cfg.BOTTLENECKS_PER_STAGE,
        decoder_type=cfg.DECODER_TYPE, use_asymmetric=cfg.USE_ASYMMETRIC, context_pattern=cfg.CONTEXT_PATTERN, separable_dilated=cfg.SEPARABLE_DILATED,
        use_prelu=g("USE_PRELU", True), prelu_variant=cfg.PRELU_VARIANT, use_dsc=g("USE_DSC", False), dsc_no_projection=g("DSC_NO_PROJECTION", False),
        dsc_no_projection_context_only=g("DSC_NO_PROJECTION_CONTEXT_ONLY", False), reg_bookend_dsc=g("REG_BOOKEND_DSC", False), dsc_separable=g("DSC_SEPARABLE", False),
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["network_weights"], strict=True)
    return model.eval()


def load_inputs(files: list[Path], device: torch.device) -> torch.Tensor:
    """(N,1,H,W) float32, z-scored exactly like the board path, resident on `device`."""
    arrs = [zscore(np.asarray(Image.open(p).convert("L"))) for p in files]
    return torch.from_numpy(np.stack(arrs)[:, None]).to(device)


# ------------------------------------------------------------------------------------------------ numeric modes
def configure_numeric_mode(device: str, gpu_mode: str) -> str:
    if device == "cpu":
        return "fp32"
    if gpu_mode == "fp32_strict":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:  # fp32_default / fp16_autocast leave PyTorch's own defaults untouched (cudnn TF32 on, matmul TF32 off)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    return gpu_mode


def forward(model, x: torch.Tensor, numeric_mode: str):
    if numeric_mode == "fp16_autocast":
        with torch.autocast("cuda", dtype=torch.float16):
            return model(x)
    return model(x)


def infer(model, x: torch.Tensor, numeric_mode: str):
    """The timed unit: network forward + per-pixel argmax (the FPGA chain ends in an argmax node and returns a class map, so the argmax is inside its timing too). The class map stays on the device."""
    return forward(model, x, numeric_mode).argmax(1)


# ------------------------------------------------------------------------------------------------ timed regions
def stream_once(model, views: list[torch.Tensor], device: str, numeric_mode: str) -> dict:
    """ONE timed region over all inputs (batch dimension taken from the pre-split views): inputs already on the device, class maps (forward + argmax) dropped on the device, no transfer inside."""
    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            w0 = time.perf_counter()
            start.record()
            for x in views:
                infer(model, x, numeric_mode)
            end.record()
            end.synchronize()
            wall = time.perf_counter() - w0
            return {"total_s": start.elapsed_time(end) / 1000.0, "wall_s": wall}
        t0 = time.perf_counter()
        for x in views:
            infer(model, x, numeric_mode)
        t = time.perf_counter() - t0
        return {"total_s": t, "wall_s": t}


def single_image_latencies(model, x: torch.Tensor, device: str, numeric_mode: str, warmup: int, n: int) -> list[float]:
    lat = []
    with torch.inference_mode():
        for i in range(warmup + n):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer(model, x, numeric_mode)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def split_views(inputs: torch.Tensor, batch: int) -> list[torch.Tensor]:
    return [inputs[i:i + batch] for i in range(0, inputs.shape[0], batch)]


# ------------------------------------------------------------------------------------------------ accuracy (never timed)
def accuracy(model, files: list[Path], inputs: torch.Tensor, numeric_mode: str) -> dict:
    labels_dir = REPO / "data" / "nnUNet_raw" / DATASET / "labelsTs"
    per_class = {c: [] for c in CLASSES}
    binary = []
    with torch.inference_mode():
        for p, x in zip(files, inputs):
            pred = infer(model, x[None], numeric_mode)[0].to("cpu", torch.uint8).numpy()
            gt = topo.load_class_id_mask(labels_dir / f"{case_id(p)}.png", binarize=False)
            pred = topo.resize_mask_to(pred, gt.shape)
            for cid, name in enumerate(CLASSES, start=1):
                per_class[name].append(topo.dice_score(gt == cid, pred == cid))
            binary.append(topo.dice_score(gt > 0, pred > 0))
    out = {f"dice_{c}": float(np.mean(per_class[c])) for c in CLASSES}
    out["dice"] = float(np.mean([out[f"dice_{c}"] for c in CLASSES]))
    out["dice_binary"] = float(np.mean(binary))
    return out


# ------------------------------------------------------------------------------------------------ environment / output
def gpu_driver() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def platform_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device == "cuda" else bench.cpu_name()


def write_env(device: str, numeric_mode: str, extra: dict) -> None:
    env = json.loads(bench.OUT_ENV.read_text()) if bench.OUT_ENV.is_file() else {}
    env[f"{device}:{numeric_mode}"] = {
        "platform": platform_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_benchmark_warm": True,
        "torch_threads": torch.get_num_threads(), "cpu": bench.cpu_name(), "gpu_driver_maxsmclk_powerlimit": gpu_driver() if device == "cuda" else None,
        "python": sys.version.split()[0], "timestamp": datetime.now().isoformat(timespec="seconds"), **extra,
    }
    bench.OUT_ENV.write_text(json.dumps(env, indent=2))


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--gpu-mode", choices=["fp32_default", "fp32_strict", "fp16_autocast"], default="fp32_default")
    ap.add_argument("--limit", type=int, help="only the first N test images (smoke test)")
    ap.add_argument("--repeats", type=int, default=5, help="stream_warm repeats in this process (mean + std reported)")
    ap.add_argument("--warmup", type=int, default=10, help="untimed warmup forwards before stream_warm / single_image")
    ap.add_argument("--single-n", type=int, default=200, help="timed forwards in single_image mode")
    ap.add_argument("--threads", type=int, help="CPU threads (default torch.get_num_threads())")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--power", action="store_true", help="also measure energy per image above idle (power_meter.py; cpu needs host_rapl_sampler.ps1 running on the Windows host)")
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--cooldown-s", type=float, default=0.0, help="idle pause before every timed stream after the first and before the single-image latencies (0 = back to back)")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available")
    if a.device == "cpu" and a.gpu_mode != "fp32_default":
        sys.exit("--gpu-mode only applies to --device cuda")
    numeric_mode = configure_numeric_mode(a.device, a.gpu_mode)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    files = bench.test_images(a.limit)
    model = build_model().to(dev)
    inputs = load_inputs(files, dev)
    n = len(files)
    views = split_views(inputs, a.batch_size)
    torch.backends.cudnn.benchmark = True

    def sync() -> None:
        if a.device == "cuda":
            torch.cuda.synchronize()

    meter = idle1 = None
    if a.power:
        import power_meter as pm
        meter = pm.make_meter(a.device)
        meter.start()
        time.sleep(5)                                               # let load / allocation activity settle, then record idle with the model resident
        idle1 = pm.idle_window(meter, a.idle_s)
    with torch.inference_mode():
        for i in range(a.warmup):
            infer(model, views[i % len(views)], numeric_mode)
        sync()
    warm_runs, windows = [], []
    for k in range(a.repeats):
        if k and a.cooldown_s:
            sync()
            time.sleep(a.cooldown_s)
        if a.power:
            windows.append([meter.mark()])
        warm_runs.append(stream_once(model, views, a.device, numeric_mode))
        if a.power:
            windows[-1].append(meter.mark())                        # energy is taken over exactly the timed streams (same thermal state as the latency row)
        print(f"stream_warm run {k + 1}/{a.repeats}: {warm_runs[-1]['total_s']:.3f} s (wall {warm_runs[-1]['wall_s']:.3f} s)", flush=True)
    energy = {}
    if a.power:
        energy = pm.summarize(meter, windows, idle1, pm.idle_window(meter, a.idle_s), n)
        meter.stop()
    if a.cooldown_s:
        time.sleep(a.cooldown_s)
    lat = single_image_latencies(model, inputs[:1], a.device, numeric_mode, a.warmup, a.single_n)
    acc = {} if a.skip_accuracy else accuracy(model, files, inputs, numeric_mode)
    if acc:
        print(f"accuracy ({n} images): dice {acc['dice']:.4f}  binary {acc['dice_binary']:.4f}  " + "  ".join(f"{c} {acc[f'dice_{c}']:.4f}" for c in CLASSES), flush=True)
        if n == 300 and abs(acc["dice"] - 0.7954) > 0.002:
            print(f"WARNING: dice {acc['dice']:.4f} differs from the recorded nnU-Net-pipeline value 0.7954 by more than 0.002", flush=True)

    plat, threads = platform_name(a.device), torch.get_num_threads()
    stamp = datetime.now().isoformat(timespec="seconds")
    base = dict(platform=plat, device=a.device, precision=PRECISION_LABEL[numeric_mode], numeric_mode=numeric_mode, model="nnUNetTrainerENet_12_dense_relu_warmstart150ep",
                batch=a.batch_size, n_images=n, threads=threads if a.device == "cpu" else "", timestamp=stamp, **{k: bench.fmt(v) for k, v in acc.items()})

    ts = [r["total_s"] for r in warm_runs]
    t, sd = statistics.fmean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)
    fps = [n / x for x in ts]
    rows = [
        {**base, "mode": "stream_warm", "runs": len(ts), "total_s": bench.fmt(t), "total_s_std": bench.fmt(sd), "latency_ms_mean": bench.fmt(t / n * 1000.0), "latency_ms_std": bench.fmt(sd / n * 1000.0),
         "throughput_fps": bench.fmt(n / t, 3), "throughput_fps_std": bench.fmt(statistics.stdev(fps) if len(fps) > 1 else 0.0, 3),
         **{k: bench.fmt(energy[k]) for k in ENERGY_ROW_KEYS if k in energy}},
        {**base, "mode": "single_image", "runs": len(lat), "latency_ms_mean": bench.fmt(statistics.fmean(lat)), "latency_ms_std": bench.fmt(statistics.stdev(lat)), "latency_ms_median": bench.fmt(statistics.median(lat)),
         "latency_ms_min": bench.fmt(min(lat)), "latency_ms_p95": bench.fmt(bench.pct(lat, 95)), "latency_ms_p99": bench.fmt(bench.pct(lat, 99))},
    ]
    bench.upsert(rows)
    write_env(a.device, numeric_mode, {"warm_runs_s": ts, "warm_runs_wall_s": [r["wall_s"] for r in warm_runs], "batch_size": a.batch_size, "warmup": a.warmup,
                                       **({"energy": {k: v for k, v in energy.items() if k not in ENERGY_ROW_KEYS}} if energy else {})})
    for r in rows:
        print({k: r[k] for k in ("mode", "runs", "total_s", "total_s_std", "latency_ms_mean", "latency_ms_std", "latency_ms_median", "latency_ms_p95", "throughput_fps", "throughput_fps_std") if r.get(k) not in (None, "")}, flush=True)
    print(f"wrote {bench.OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
