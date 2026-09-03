#!/bin/bash
set -euo pipefail

# Run this ONCE before submitting upscale_pareto_array.job or
# upscale_graduate_array.job. Those are 45-task / N-task Slurm ARRAYS --
# if each task does its own "preprocess if missing" check, many of them
# start before the first one finishes, race to write
# nnUNet_preprocessed/Dataset501_ARCADE/ concurrently, and you get
# FileNotFoundError / corrupt .pkl errors deep in nnUNetDataLoader2D.
# Preprocessing once, up front, as a plain non-array job sidesteps the
# race entirely -- by the time the array starts, nnUNetPlans.json already
# exists and every task's own "if missing" check is a same no-op read.
#
# Usage: bash upscale/slurm/preprocess_501.sh

REPO_ROOT="$HOME/LightM-UNet"

echo "=== Job info ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

module purge
module load 2023

source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u
conda activate "$HOME/conda-envs/lightmunet"
set -u

unset PYTHONPATH
export PYTHONHASHSEED=42

export nnUNet_raw="$REPO_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$REPO_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$REPO_ROOT/data/nnUNet_results"

PLANS_FILE="$nnUNet_preprocessed/Dataset501_ARCADE/nnUNetPlans.json"
if [ -f "$PLANS_FILE" ]; then
    echo "=== $PLANS_FILE already exists -- skipping preprocessing ==="
else
    echo "=== Preprocessing Dataset501_ARCADE ==="
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
fi

if [ -f "$nnUNet_raw/Dataset501_ARCADE/splits_final.json" ]; then
    cp "$nnUNet_raw/Dataset501_ARCADE/splits_final.json" \
       "$nnUNet_preprocessed/Dataset501_ARCADE/splits_final.json"
    echo "=== Synced splits_final.json ==="
fi

echo "Done. Safe to submit upscale_pareto_array.job / upscale_graduate_array.job now."
