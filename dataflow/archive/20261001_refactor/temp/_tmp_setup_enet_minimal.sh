#!/usr/bin/env bash
set -e
echo "=== System libs ==="
apt-get update -qq
apt-get install -y -qq libxcb1 libx11-6 libxext6 libgl1 libglib2.0-0 >/dev/null

echo "=== Python deps (minimal set for preprocessing + FINN export, no Mamba build) ==="
python -m pip install --upgrade pip -q
python -m pip install -q \
    numpy==1.26.4 \
    scipy \
    scikit-image \
    scikit-learn \
    batchgenerators \
    acvl-utils \
    dynamic-network-architectures \
    dicom2nifti \
    medpy \
    SimpleITK \
    tqdm \
    matplotlib \
    seaborn \
    pandas \
    graphviz \
    tifffile \
    requests \
    nibabel \
    imagecodecs \
    yacs \
    monai==1.3.0 \
    hiddenlayer \
    thop \
    importlib_metadata \
    importlib_resources \
    brevitas==0.12.1 \
    qonnx \
    onnx \
    onnxscript \
    onnxoptimizer \
    pulp

python -m pip uninstall -y -q opencv-python opencv-python-headless || true
python -m pip install -q opencv-python-headless==4.9.0.80

echo "=== Editable install of enet (nnunetv2) ==="
cd /workspace/LightM-UNet/enet
python -m pip install -q -e . --no-deps

echo "=== Verifying imports ==="
python3 -c "
import nnunetv2, brevitas, qonnx, batchgenerators, acvl_utils, dynamic_network_architectures
print('nnunetv2', nnunetv2.__file__)
print('brevitas', brevitas.__version__)
print('OK')
"
