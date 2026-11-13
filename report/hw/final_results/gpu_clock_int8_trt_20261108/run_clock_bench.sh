#!/bin/bash
# INT8 TensorRT GPU benchmark with a 50 Hz NVML clock/power/temperature sampler running next to it (inside lightmunet_dev).
set -o pipefail
OUT=/tmp/clockrun
rm -rf "$OUT"; mkdir -p "$OUT"
export PYTHONPATH=/opt/trt_pkgs
export LD_LIBRARY_PATH=/opt/trt_pkgs/tensorrt_libs:/opt/trt_pkgs/nvidia/cuda_runtime/lib:/opt/trt_pkgs/nvidia/cublas/lib:/opt/trt_pkgs/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
cd /workspace/LightM-UNet

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,temperature.gpu,pstate --format=csv > "$OUT/gpu_before.csv"
python3 /tmp/clock_sampler.py "$OUT/clocks.csv" 0.02 &
SP=$!
sleep 3                                                         # idle baseline, model not loaded yet
echo "$(date +%s.%N) START" > "$OUT/run.log"
python3 -u report/hw/benchmark/benchmark_int8_cpu_gpu.py --backend trt-gpu "$@" 2>&1 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done | tee -a "$OUT/run.log"
RC=${PIPESTATUS[0]}
echo "$(date +%s.%N) END rc=$RC" >> "$OUT/run.log"
sleep 3                                                         # idle tail
kill -TERM $SP; wait $SP
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,pstate --format=csv > "$OUT/gpu_after.csv"
exit $RC
