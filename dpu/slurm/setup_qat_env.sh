#!/usr/bin/env bash
# One-off: build a conda env on the cluster with Vitis AI 3.0 vai_q_pytorch (pytorch_nndct) + CUDA kernels,
# so qat_train.py can run QAT on a GPU. The Vitis AI docker image's own pytorch_nndct is CPU-only.
#
#   VITIS_AI_SRC=$HOME/Vitis-AI bash setup_qat_env.sh
#
# vai_q_pytorch supports python 3.6-3.9 and torch 1.2-1.12; this pins python 3.8 + torch 1.12.1+cu116.
# Status: written from the Vitis AI source, NOT yet run on the cluster. If the nvcc/torch CUDA-version
# check or the GPU arch below fails on your nodes, that is where to adjust.
set -euo pipefail

VITIS_AI_SRC="${VITIS_AI_SRC:-$HOME/Vitis-AI}"
ENV_DIR="${ENV_DIR:-$HOME/conda-envs/vitisqat}"
CUDA_ARCHS="${CUDA_ARCHS:-80}"        # A100 = 80; add 86/90 as needed (torch 1.12 has no native sm_90)
SRC="$VITIS_AI_SRC/src/vai_quantizer/vai_q_pytorch"
[[ -d "$SRC" ]] || { echo "missing $SRC (copy Vitis-AI/src/vai_quantizer to the cluster)" >&2; exit 1; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -y -p "$ENV_DIR" python=3.8
conda activate "$ENV_DIR"

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# nvcc 11.x must match torch's CUDA major version (the cluster's CUDA 12 module would be rejected).
conda install -y -c "nvidia/label/cuda-11.6.2" cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvrtc-dev
pip install "numpy<1.24" "scipy<1.11" "scikit-learn<1.3" tqdm ninja six tabulate graphviz==0.19.1 \
    PyYAML networkx==2.5.1 "batchgenerators==0.25" pandas SimpleITK

# Build from a scratch copy: setup.py symlinks nndct_shared and hardcodes -arch=sm_35 (dropped in CUDA 12, too old for A100).
BUILD="$(mktemp -d)"
cp -r "$SRC" "$BUILD/vai_q_pytorch"
cd "$BUILD/vai_q_pytorch/pytorch_binding"
GENCODE=""
for a in $CUDA_ARCHS; do GENCODE="$GENCODE'-gencode','arch=compute_$a,code=sm_$a',"; done
sed -i "s/\['-O2','-arch=sm_35'\]/['-O2',${GENCODE%,}]/" setup.py
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHS" | sed 's/\([0-9]\)\([0-9]\)/\1.\2/g')"
python setup.py install

python - <<'E'
import torch, pytorch_nndct
print("torch", torch.__version__, "cuda", torch.version.cuda, "nndct", pytorch_nndct.__file__)
E
