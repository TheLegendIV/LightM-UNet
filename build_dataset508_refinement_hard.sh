#!/bin/bash
set -euo pipefail

# Build Dataset508_ARCADE_refinement_hard (dataset-prep/prepare_arcade_508_refinement_hard.py)
# from Dataset501_ARCADE's raw images/GT plus nnUNetTrainerENetOriginal's
# saved probability output. Plain local script, not a SLURM job -- pure
# CPU/numpy work (skeletonize, connected components, farthest-point
# sampling) over 1200 train+val + 300 test 512x512 images, no GPU. Local
# dry-run projection (same box-finding logic, against the existing
# binarized predictions as a stand-in) processed all 1200 train+val images
# in a few minutes on a single CPU core -- light enough to just run
# directly on an HPC node instead of queuing it.
#
# REQUIRES run_enetoriginal_probabilities.job to have completed first --
# this reads Dataset501_ARCADE/labelsPr_ENetOriginal_Tr/*.npz and
# Dataset501_ARCADE/labelsPr_ENetOriginal/*.npz (vessel probability =
# 1 - probabilities[background]), which only exist after that job's
# --save_probabilities run; the older labelsPr_ENetOriginal_Tr/
# labelsPr_ENetOriginal only have binarized .png labels and will make this
# fail the check below.
#
# Output: data/nnUNet_raw/Dataset508_ARCADE_refinement_hard/ -- imagesTr/
# imagesTs (each case: {id}_0000.png raw grayscale, {id}_0001.png vessel
# probability as uint8), labelsTr/labelsTs (GT), dataset.json,
# splits_final.json. NOT preprocessed yet -- run preprocess_dataset508.job
# next.
#
# Usage: bash build_dataset508_refinement_hard.sh

REPO_ROOT="$HOME/LightM-UNet"

echo "=== Job info ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

module purge
module load 2023

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda's package-level activation hooks (e.g. the CUDA nvcc one, which
# references NVCC_PREPEND_FLAGS with no default) aren't written to be
# nounset-safe -- disable -u just around activation, then restore it.
set +u
conda activate "$HOME/conda-envs/lightmunet"
set -u

unset PYTHONPATH
export PYTHONHASHSEED=42

echo "=== Verify probability .npz files exist (run_enetoriginal_probabilities.job prerequisite) ==="
TR_PRED_DIR="$REPO_ROOT/data/nnUNet_raw/Dataset501_ARCADE/labelsPr_ENetOriginal_Tr"
TS_PRED_DIR="$REPO_ROOT/data/nnUNet_raw/Dataset501_ARCADE/labelsPr_ENetOriginal"
TR_NPZ_COUNT=$(ls "$TR_PRED_DIR"/*.npz 2>/dev/null | wc -l)
TS_NPZ_COUNT=$(ls "$TS_PRED_DIR"/*.npz 2>/dev/null | wc -l)
echo "labelsPr_ENetOriginal_Tr .npz count: ${TR_NPZ_COUNT}"
echo "labelsPr_ENetOriginal .npz count: ${TS_NPZ_COUNT}"
if [ "$TR_NPZ_COUNT" -eq 0 ] || [ "$TS_NPZ_COUNT" -eq 0 ]; then
    echo "ERROR: no .npz probability files found -- run run_enetoriginal_probabilities.job first." >&2
    exit 1
fi

echo "=== Building Dataset508_ARCADE_refinement_hard ==="
python "$REPO_ROOT/dataset-prep/prepare_arcade_508_refinement_hard.py"

echo "=== Case counts by category ==="
OUT_DIR="$REPO_ROOT/data/nnUNet_raw/Dataset508_ARCADE_refinement_hard"
for split in labelsTr labelsTs; do
    echo "-- ${split} --"
    ls "$OUT_DIR/${split}" | grep -oE '_(discont|corrupt|empty)_' | sort | uniq -c
done

echo "Done. Dataset508 build finished. Ready for preprocess_dataset508.job."
