#!/bin/bash
#SBATCH --job-name=preprocess_509
#SBATCH --partition=rome           # CPU-only -- no GPU needed for plan_and_preprocess; check `sinfo -o "%P %G"`
                                   # for this cluster's actual CPU-only partition name, this is a guess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Plan+preprocess Dataset509_ARCADE_1x1_4c from nnUNet_raw only -- no
# training, no GPU. Every stage_1/2/3/4 job under compression/slurm/ still
# has its own inline "preprocess if not already done" guard, so running
# this first is optional, not required for a single job -- but it matters
# for stage_1_naive_baseline_array.job (5 tasks), stage_2_special_ops_array.job
# (4 tasks), and stage_4_arch_probes_array.job (9 tasks): if SLURM starts
# several array tasks at once, they can all see "no plans yet" simultaneously
# and race to run nnUNetv2_plan_and_preprocess concurrently, corrupting the
# shared nnUNet_preprocessed/Dataset509_ARCADE_1x1_4c/ cache (FileNotFoundError
# / corrupt .pkl errors deep in nnUNetDataLoader2D -- same failure mode
# upscale/slurm/preprocess_501.sh was written to avoid for that dataset).
# Also keeps the ~20-40 min, CPU/IO-bound preprocessing step off a scarce
# gpu_mig slot while nothing on the GPU side is happening yet. One run of
# this covers every stage job -- plans.json is per dataset+configuration,
# not per trainer, so nnUNetTrainerENet (the single generic trainer every
# stage_1/2/3/4 job uses, distinguished only by ENET_* env vars) reads the
# same preprocessed cache regardless of which stage/config runs.
#
# Preprocessed output is NOT committed to git (data/nnUNet_preprocessed/*
# is gitignored -- regenerated on demand) -- this script is how you
# regenerate it on a fresh checkout before submitting any stage_*.job.



DATASET_ID=509
DATASET_NAME="Dataset509_ARCADE_1x1_4c"

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

echo "=== Plan + preprocess ==="
if [ ! -f "$nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans.json" ]; then
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity
else
    echo "Plans already exist at $nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans.json, skipping."
fi

echo "=== Ensure official split is present ==="
cp "$nnUNet_raw/${DATASET_NAME}/splits_final.json" \
   "$nnUNet_preprocessed/${DATASET_NAME}/splits_final.json"

echo "Preprocessing finished for ${DATASET_NAME}. Ready for stage_1_naive_baseline_array.job / stage_2_special_ops_array.job / stage_3_transfer_original.job / stage_4_arch_probes_array.job."
