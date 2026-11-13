#!/bin/bash
# Full CPU + GPU benchmark set with clock logging (inside lightmunet_dev; repo at /workspace/LightM-UNet). usage: run_benchmarks_with_clocks.sh <out_dir> [gap_s]
# Six runs, each a complete benchmark process (--power: energy + the host sampler flag): GPU runs with the NVML sampler (gpu_clock_sampler.py) around them, CPU runs relying on the
# Windows host's host_rapl_sampler.ps1 + host_cpu_clock_sampler.ps1 (started beforehand; <out_dir>/cpu_clock.csv). Per run: <out_dir>/<tag>/run.log (every output line stamped with the container
# clock: stream_warm windows = stamp - wall .. stamp) and, for GPU runs, gpu_clock.csv. summarize_benchmark_clocks.py turns these into clock_summary.csv.
# Each run upserts its rows in report/hw/final_results/benchmark_s12_512.csv and writes benchmark_s12_512_env.json.
set -o pipefail
OUT="${1:?out_dir}"
GAP="${2:-60}"                                                    # idle pause between runs so each starts from a similar thermal state
cd /workspace/LightM-UNet
mkdir -p "$OUT"
TRT_ENV="PYTHONPATH=/opt/trt_pkgs LD_LIBRARY_PATH=/opt/trt_pkgs/tensorrt_libs:/opt/trt_pkgs/nvidia/cuda_runtime/lib:/opt/trt_pkgs/nvidia/cublas/lib:/opt/trt_pkgs/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
B=report/hw/benchmark

# tag | device | env prefix | command
RUNS=(
  "gpu_fp32_default|gpu||python3 -u $B/benchmark_fp32_cpu_gpu.py --device cuda --power"
  "gpu_fp32_strict|gpu||python3 -u $B/benchmark_fp32_cpu_gpu.py --device cuda --gpu-mode fp32_strict --power"
  "gpu_fp16_autocast|gpu||python3 -u $B/benchmark_fp32_cpu_gpu.py --device cuda --gpu-mode fp16_autocast --power"
  "gpu_int8_trt|gpu|$TRT_ENV|python3 -u $B/benchmark_int8_cpu_gpu.py --backend trt-gpu --power"
  "cpu_fp32|cpu||python3 -u $B/benchmark_fp32_cpu_gpu.py --device cpu --threads 8 --power"
  "cpu_int8_ort|cpu||python3 -u $B/benchmark_int8_cpu_gpu.py --backend ort-cpu --power"
)

n=${#RUNS[@]}; i=0
for spec in "${RUNS[@]}"; do
  i=$((i + 1))
  IFS='|' read -r TAG DEV ENVP CMD <<< "$spec"
  D="$OUT/$TAG"; rm -rf "$D"; mkdir -p "$D"
  echo "=== [$i/$n] $TAG ($(date +%H:%M:%S))"
  SP=""
  if [ "$DEV" = gpu ]; then
    python3 "$B/gpu_clock_sampler.py" "$D/gpu_clock.csv" 0.02 & SP=$!
    sleep 3                                                       # idle baseline before the process starts
  fi
  echo "$(date +%s.%N) START" > "$D/run.log"
  env $ENVP $CMD 2>&1 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done | tee -a "$D/run.log"
  RC=${PIPESTATUS[0]}
  echo "$(date +%s.%N) END rc=$RC" >> "$D/run.log"
  if [ -n "$SP" ]; then sleep 3; kill -TERM $SP; wait $SP 2>/dev/null; fi
  echo "=== $TAG finished rc=$RC"
  if [ $i -lt $n ]; then sleep "$GAP"; fi
done
echo "=== all runs done ($(date +%H:%M:%S))"
