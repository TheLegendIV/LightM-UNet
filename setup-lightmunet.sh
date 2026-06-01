#!/usr/bin/env bash
set -e

echo "=== Installing system libraries for OpenCV/headless compatibility ==="
apt-get update
apt-get install -y libxcb1 libx11-6 libxext6 libgl1 libglib2.0-0

echo "=== Checking PyTorch/CUDA from Docker image ==="
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA is not available in PyTorch"
PY

echo "=== Upgrading pip tooling ==="
python -m pip install --upgrade pip
python -m pip install \
    setuptools==80.9.0 \
    wheel==0.47.0 \
    packaging==26.2 \
    ninja==1.13.0

echo "=== Installing base Python dependencies ==="
python -m pip install \
    numpy==1.26.4 \
    transformers==4.35.2 \
    tokenizers==0.15.2 \
    scipy \
    scikit-image \
    batchgenerators \
    acvl-utils \
    tqdm \
    matplotlib \
    pandas \
    hiddenlayer \
    importlib_metadata \
    importlib_resources

echo "=== Installing OpenCV headless ==="
python -m pip uninstall -y opencv-python opencv-python-headless || true
python -m pip install opencv-python-headless==4.9.0.80

echo "=== Removing conflicting Mamba packages ==="
python -m pip uninstall -y mamba mamba-ssm causal-conv1d tilelang tvm tvm-ffi torch-c-dlpack-ext || true

echo "=== Installing Mamba CUDA dependencies ==="
export MAX_JOBS=4

MAX_JOBS=4 python -m pip install causal-conv1d==1.1.1 --no-build-isolation
MAX_JOBS=4 python -m pip install mamba-ssm==1.1.1 --no-build-isolation

echo "=== Testing imports before editable install ==="
python - <<'PY'
import torch
import numpy
import transformers
import cv2
import mamba_ssm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("cv2:", cv2.__version__)
print("mamba_ssm OK")

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.__version__.startswith("2.0.1"), torch.__version__
assert torch.version.cuda == "11.7", torch.version.cuda
assert numpy.__version__.startswith("1.26"), numpy.__version__
assert transformers.__version__ == "4.35.2", transformers.__version__
assert cv2.__version__.startswith("4.9"), cv2.__version__
PY

echo "=== Installing LightM-UNet editable package ==="

python -m pip install -e .

python -m pip install numpy==1.26.4 --force-reinstall
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --no-cache-dir opencv-python-headless==4.9.0.80

echo "=== Final test ==="
python - <<'PY'
import torch
import numpy
import transformers
import cv2
import mamba_ssm
import nnunetv2

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("cv2:", cv2.__version__)
print("mamba_ssm OK")
print("nnunetv2 OK")
PY

echo "Done."