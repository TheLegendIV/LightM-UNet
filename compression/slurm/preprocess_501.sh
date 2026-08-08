#!/bin/bash
#SBATCH --job-name=preprocess_501
#SBATCH --partition=rome           # CPU-only -- no GPU needed for plan_and_preprocess; check `sinfo -o "%P %G"`
                                   # for this cluster's actual CPU-only partition name, this is a guess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Plan+preprocess Dataset501_ARCADE (binary vessel segmentation, 1x1 grid --
# full-size images, 1000 train / 200 val / 300 test) from nnUNet_raw only --
# no training, no GPU. Same structure as this folder's own preprocess_509.sh
# (Dataset509_ARCADE_1x1_4c's preprocessing job); this dataset was retired
# in favor of Dataset509's 4-class successor earlier in the project (see
# compression/slurm/archive/README.md) and is being brought back
# specifically as a PRETRAINING source -- binary vessel-presence is an
# easier task than 4-class branch identification, so a checkpoint trained
# here is a candidate warm start for a 4-class model's encoder, the same
# way S13 was warm-started from S5-SeparableDense's own (same-dataset)
# checkpoint. nnUNet_raw/Dataset501_ARCADE was rebuilt via
# `python dataset-prep/prepare_arcade.py --dataset-id 501 --source syntax
# --grid 1x1 --name Dataset501_ARCADE` -- deterministic from the ARCADE
# source COCO annotations (train/val/test splits come from that source's
# own folder structure, not a random split), confirmed to reproduce the
# original dataset.json (channel_names/labels/numTraining=1200/file_ending)
# byte-for-byte (whitespace aside) against the copy embedded in this
# dataset's existing old checkpoints under data/nnUNet_results/Dataset501_ARCADE/.
#
# Preprocessed output is NOT committed to git (data/nnUNet_preprocessed/*
# is gitignored -- regenerated on demand) -- this script is how you
# regenerate it on a fresh checkout before submitting any job that trains
# against Dataset501_ARCADE.

DATASET_ID=501
DATASET_NAME="Dataset501_ARCADE"

# -- HPC environment ----------------------------------------------------------
module purge
module load 2023

# -- Activate Conda environment -----------------------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/lightmunet"

# -- General settings ---------------------------------------------------------
unset PYTHONPATH
export PYTHONHASHSEED=42

# -- nnU-Net paths ------------------------------------------------------------
export nnUNet_raw="$HOME/LightM-UNet/data/nnUNet_raw"
export nnUNet_preprocessed="$HOME/LightM-UNet/data/nnUNet_preprocessed"
export nnUNet_results="$HOME/LightM-UNet/data/nnUNet_results"

cd "$HOME/LightM-UNet/enet"

echo "=== Job info ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Dataset: ${DATASET_NAME} (id ${DATASET_ID})"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"

echo "=== Checking nnUNet_raw/${DATASET_NAME} exists ==="
if [ ! -f "$nnUNet_raw/${DATASET_NAME}/dataset.json" ]; then
    echo "ERROR: $nnUNet_raw/${DATASET_NAME}/dataset.json not found -- rebuild the raw dataset first:" >&2
    echo "  python dataset-prep/prepare_arcade.py --dataset-id 501 --source syntax --grid 1x1 --name Dataset501_ARCADE" >&2
    exit 1
fi

echo "=== Plan + preprocess ==="
if [ ! -f "$nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans.json" ]; then
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity
else
    echo "Plans already exist at $nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans.json, skipping."
fi

echo "=== Ensure official split is present ==="
cp "$nnUNet_raw/${DATASET_NAME}/splits_final.json" \
   "$nnUNet_preprocessed/${DATASET_NAME}/splits_final.json"

echo "Preprocessing finished for ${DATASET_NAME}. Ready for a pretraining job against Dataset501_ARCADE."
