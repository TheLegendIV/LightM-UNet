TensorRT for the lightmunet_dev container (torch 2.0.1 + CUDA 11.7 stays untouched; TensorRT 10.3 with CUDA 12.2 runtime libs lives in /opt/trt_pkgs, found through PYTHONPATH / LD_LIBRARY_PATH, see run_trt_int8.sh):

    docker exec lightmunet_dev pip install --no-cache-dir --target /opt/trt_pkgs --no-deps tensorrt-cu12==10.3.0 tensorrt-cu12-libs==10.3.0 tensorrt-cu12-bindings==10.3.0
    docker exec lightmunet_dev pip install --no-cache-dir --target /opt/trt_pkgs --no-deps nvidia-cuda-runtime-cu12==12.2.140 nvidia-cublas-cu12==12.2.5.6 nvidia-cuda-nvrtc-cu12==12.2.140

(/opt is inside the container: reinstall after recreating it.) Then:  python analysis/benchmark/export_int8_qcdq.py ; bash analysis/benchmark/run_trt_int8.sh ; python analysis/benchmark/benchmark_int8_cpu_gpu.py --backend ort-cpu
The TensorRT engine (int8/*.plan, ~6 min to build) is cached next to the ONNX files; delete it after re-exporting the ONNX.
