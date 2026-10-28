export PYTHONPATH=/opt/trt_pkgs
export LD_LIBRARY_PATH=/opt/trt_pkgs/tensorrt_libs:/opt/trt_pkgs/nvidia/cuda_runtime/lib:/opt/trt_pkgs/nvidia/cublas/lib:/opt/trt_pkgs/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
cd /workspace/LightM-UNet
python3 analysis/benchmark/benchmark_int8_cpu_gpu.py --backend trt-gpu "$@"
