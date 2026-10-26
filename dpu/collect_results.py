"""Benchmark the compiled DPU model on the ZCU104 (2x B4096) and upsert Dice / speed / energy rows into the shared
benchmark table analysis/report/MILP/final_results/benchmark_s12_512.csv (same schema, modes and timing rules as
analysis/benchmark/benchmark_int8_cpu_gpu.py and the FPGA row).

  python dpu/collect_results.py [--job 12_dense_relu_warmstart150ep_fp32] [--threads 2] [--repeats 5]
  python dpu/collect_results.py --no-run     # re-score the stored board outputs, keep the stored timing / power

Run on the Windows host (needs numpy, Pillow, scipy; ssh/scp go through WSL, where the board key is installed).
Prerequisite: `./run_flow.sh <job> compile` (the compiled xmodel ends in an argmax on the DPU).

Per run (dpu/bench_board.py on the board):
  timed unit    DPU input -> DPU output, class map (argmax on the DPU) already in memory. Preprocessing (z-score,
                quantization with the DPU input scale) is done for all 300 test images before the first timed run and
                is excluded, as are the host transfers.
  stream_warm   --warmup untimed inferences, then --repeats passes over the 300 images, batch 1, --threads VART runners
                (one per DPU core). latency_ms_mean = total_s / n (time per image of the stream, as in the other rows).
  single_image  --single-n synchronous runs of one image on one runner (mean/std/median/min/p95/p99).
  energy        INA226 on the 12 V board input, sampled at ~200 Hz (the sensor updates at ~114 Hz); idle windows with the
                model loaded before and after the streams, energy over exactly the timed streams, dynamic energy above idle
                (analysis/benchmark/power_meter.summarize). WHOLE BOARD at the input (PS + PL + DDR + regulator losses + fan),
                not the post-regulator rail sum the FINN ZCU106 power rows use.
  accuracy      untimed pass over all test images, same Dice definitions as deployment/collect_results.py.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "analysis" / "benchmark"))
import benchmark_common as bench  # noqa: E402
import power_meter as pm  # noqa: E402
from img_intf.utils.preprocess import case_id  # noqa: E402

PLATFORM = "Zynq UltraScale+ ZCU104 (DPU, 2xB4096)"
NUMERIC_MODE = "int8_dpu"
NUMERIC_MODE_QAT = "int8_dpu_qat"
BOARD_DIR = "/home/root/dpu_bench"


def wsl_path(p: Path) -> str:
    p = p.resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def wsl(cmd: str, check=True) -> None:
    r = subprocess.run(["wsl.exe", "--exec", "bash", "-lc", cmd])
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")


def job_env(job: str) -> dict:
    env = {}
    for line in (HERE / "builds" / job / "job.env").read_text().splitlines():
        m = re.match(r"^([A-Z_]+)=([^\s#]+)", line)
        if m:
            env[m.group(1)] = m.group(2)
    return env


class BoardMeter:
    """Energy from the board-side power samples (power_meter.summarize interface)."""

    name = "ina226-12v"

    def __init__(self, samples, info):
        self.t, self.w, self._info = samples["t"], samples["w"], info

    def energy_j(self, t0, t1):
        t, w = self.t, self.w
        if t0 < t[0] or t1 > t[-1]:
            raise RuntimeError("energy window outside the sampled range")

        def at(x):
            i = max(1, bisect.bisect_left(t, x))
            f = (x - t[i - 1]) / (t[i] - t[i - 1])
            return w[i - 1] + f * (w[i] - w[i - 1])

        i0, i1 = bisect.bisect_right(t, t0), bisect.bisect_left(t, t1)
        xs, ys = [t0] + t[i0:i1] + [t1], [at(t0)] + w[i0:i1] + [at(t1)]
        return sum((xs[k + 1] - xs[k]) * (ys[k + 1] + ys[k]) / 2 for k in range(len(xs) - 1))

    def info(self):
        return {"energy_source": self._info["power_source"], "n_power_samples": len(self.t),
                "ina226_sampling_frequency_hz": self._info.get("ina226_sampling_frequency_hz"),
                "ina226_oversampling_ratio": self._info.get("ina226_oversampling_ratio")}


def accuracy(pred, files):
    import segmentation_topology as topo
    labels_dir = REPO / "data" / "nnUNet_raw" / bench.DATASET / "labelsTs"
    per_class = {c: [] for c in bench.CLASSES}
    binary = []
    for i, p in enumerate(files):
        gt = topo.load_class_id_mask(labels_dir / f"{case_id(p)}.png", binarize=False)
        pr = topo.resize_mask_to(pred[i], gt.shape)
        for cid, name in enumerate(bench.CLASSES, start=1):
            per_class[name].append(topo.dice_score(gt == cid, pr == cid))
        binary.append(topo.dice_score(gt > 0, pr > 0))
    out = {f"dice_{c}": float(np.mean(per_class[c])) for c in bench.CLASSES}
    out["dice"] = float(np.mean([out[f"dice_{c}"] for c in bench.CLASSES]))
    out["dice_binary"] = float(np.mean(binary))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default="12_dense_relu_warmstart150ep_fp32")
    ap.add_argument("--board-ip")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--single-n", type=int, default=200)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="skip the board run, re-score the stored outputs")
    ap.add_argument("--qat", action="store_true", help="benchmark the QAT model (compiled_qat/, run_flow.sh qat_compile)")
    a = ap.parse_args()
    name = f"{a.job}_qat" if a.qat else a.job
    numeric_mode = NUMERIC_MODE_QAT if a.qat else NUMERIC_MODE

    ip = a.board_ip or job_env(a.job).get("BOARD_IP")
    if not ip:
        raise SystemExit("BOARD_IP not set (job.env or --board-ip)")
    files = bench.test_images(a.limit)
    n = len(files)
    cache, out_dir = HERE / "cache", HERE / "results" / name
    cache.mkdir(exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = cache / f"test_u8_{n}.npy"

    if not a.no_run:
        if not stack.is_file():
            np.save(stack, np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in files]))
        ssh = f"ssh -o BatchMode=yes root@{ip}"
        model = f"$HOME/dpu_work/{a.job}/{'compiled_qat' if a.qat else 'compiled'}/{name}.xmodel"
        wsl(f"{ssh} 'mkdir -p {BOARD_DIR}' && scp -O {model} {wsl_path(HERE / 'bench_board.py')} root@{ip}:{BOARD_DIR}/")
        wsl(f"{ssh} 'test $(stat -c %s {BOARD_DIR}/{stack.name} 2>/dev/null || echo 0) -eq {stack.stat().st_size}' "
            f"|| scp -O {wsl_path(stack)} root@{ip}:{BOARD_DIR}/")
        power = "--no-power" if a.no_power else f"--idle-s {a.idle_s}"
        wsl(f"{ssh} 'cd {BOARD_DIR} && rm -rf out && python3 bench_board.py --model {name}.xmodel --images {stack.name} "
            f"--threads {a.threads} --repeats {a.repeats} --warmup {a.warmup} --single-n {a.single_n} {power} --out out'")
        wsl(f"scp -O 'root@{ip}:{BOARD_DIR}/out/*' {wsl_path(out_dir)}/")

    res = json.loads((out_dir / "result.json").read_text())
    pred = np.load(out_dir / "pred.npy")
    if len(pred) != n:
        raise SystemExit(f"stored outputs have {len(pred)} images, expected {n}: rerun without --no-run")
    acc = accuracy(pred, files)
    print(f"accuracy ({n} images): dice {acc['dice']:.4f}  binary {acc['dice_binary']:.4f}  "
          + "  ".join(f"{c} {acc[f'dice_{c}']:.4f}" for c in bench.CLASSES), flush=True)

    energy = {}
    if res["power_file"]:
        samples = json.loads((out_dir / res["power_file"]).read_text())
        meter = BoardMeter(samples, res["info"])
        energy = pm.summarize(meter, [(r["t0"], r["t1"]) for r in res["runs"]], res["idle1"], res["idle2"], n)

    ts = [r["total_s"] for r in res["runs"]]
    lat = res["latency_ms"]
    t, sd = statistics.fmean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)
    fps = [n / v for v in ts]
    ts_now = datetime.now().isoformat(timespec="seconds")
    base = dict(platform=PLATFORM, device="dpu", precision=f"INT8 ({'QAT' if a.qat else 'PTQ'}, Vitis AI)", numeric_mode=numeric_mode,
                model=f"{name} (argmax on DPU)", batch=1, n_images=n, threads=res["threads"], timestamp=ts_now,
                **{k: bench.fmt(v) for k, v in acc.items()})
    ekeys = ("energy_dyn_j_per_image", "energy_j_per_image", "power_active_w", "power_idle_w")
    rows = [
        {**base, "mode": "stream_warm", "runs": len(ts), "total_s": bench.fmt(t), "total_s_std": bench.fmt(sd),
         "latency_ms_mean": bench.fmt(t / n * 1000.0), "latency_ms_std": bench.fmt(sd / n * 1000.0),
         "throughput_fps": bench.fmt(n / t, 3), "throughput_fps_std": bench.fmt(statistics.stdev(fps) if len(fps) > 1 else 0.0, 3),
         **{k: bench.fmt(energy[k]) for k in ekeys if k in energy}},
        {**base, "mode": "single_image", "runs": len(lat), "threads": 1, "latency_ms_mean": bench.fmt(statistics.fmean(lat)),
         "latency_ms_std": bench.fmt(statistics.stdev(lat)), "latency_ms_median": bench.fmt(statistics.median(lat)),
         "latency_ms_min": bench.fmt(min(lat)), "latency_ms_p95": bench.fmt(bench.pct(lat, 95)), "latency_ms_p99": bench.fmt(bench.pct(lat, 99))},
    ]
    bench.upsert(rows)
    env = json.loads(bench.OUT_ENV.read_text()) if bench.OUT_ENV.is_file() else {}
    env[f"dpu:{numeric_mode}"] = {
        "platform": PLATFORM, "job": a.job, "input": "int8 NHWC, z-score + DPU input fix-point scale, resident before timing",
        "input_scale": res["input_scale"], "warm_runs_s": ts, "warmup": res["warmup"], "batch_size": 1, "timestamp": ts_now,
        "timed_unit": "DPU execute -> class map after on-DPU argmax (preprocessing excluded)", **res["info"],
        **({"energy": {k: v for k, v in energy.items() if k not in ekeys}} if energy else {})}
    bench.OUT_ENV.write_text(json.dumps(env, indent=2))
    for r in rows:
        print({k: r[k] for k in ("mode", "runs", "total_s", "total_s_std", "latency_ms_mean", "latency_ms_std", "latency_ms_median",
                                 "throughput_fps", "energy_j_per_image", "energy_dyn_j_per_image", "power_active_w", "power_idle_w")
               if r.get(k) not in (None, "")}, flush=True)
    print(f"wrote {bench.OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
