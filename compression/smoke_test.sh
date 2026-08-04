#!/usr/bin/env bash
# Pre-flight smoke test (smoke_test_policy in agent_instructions_1.yaml):
# a fast end-to-end pipeline check before a stage's full 150-epoch sweep,
# not a single upfront Stage-0 gate. Also doubles as the training-time
# estimator: run once on the U16 (floor) config with no batch-size override
# (see below) to get a per-epoch wall-clock number, then use
# report_run_estimate.sh-style arithmetic (n_runs * epoch_time * 150) before
# submitting any Slurm array.
#
# Run from the repo root, inside the training environment (a local docker
# container per setup-enet.sh, or the HPC conda env): bash compression/smoke_test.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="${1:-smoke_test_U16}"
ENET_CHANNELS_VAL="${2:-4,4,8,4,4}"           # U16 (floored) config by default -- see stage_4_arch_probes_array.job's header
ENET_BOTTLENECKS_VAL="${3:-4,8,8,2,1}"        # ENet-native depth
# upsample_conv has no channel-symmetry constraint (unlike max_unpool's
# MaxUnpool2d, which needs indices' channel count to match -- see ENet.py's
# self-test / symmetric() check), so it's the correct default here.
ENET_DECODER_TYPE_VAL="${4:-upsample_conv}"
ENET_USE_DILATED_VAL="${5:-1}"
ENET_USE_ASYMMETRIC_VAL="${6:-1}"
ENET_USE_STRIDED_VAL="${7:-1}"

DATASET_NAME="Dataset509_ARCADE_1x1_4c"
DATASET_ID=509

export nnUNet_raw="$REPO_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$REPO_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$REPO_ROOT/data/nnUNet_results"

export ENET_CHANNELS="$ENET_CHANNELS_VAL"
export ENET_BOTTLENECKS="$ENET_BOTTLENECKS_VAL"
export ENET_DECODER_TYPE="$ENET_DECODER_TYPE_VAL"
export ENET_USE_DILATED="$ENET_USE_DILATED_VAL"
export ENET_USE_ASYMMETRIC="$ENET_USE_ASYMMETRIC_VAL"
export ENET_USE_STRIDED="$ENET_USE_STRIDED_VAL"
export ENET_EPOCHS=3
export ENET_ITERATIONS_PER_EPOCH=2
export ENET_VAL_ITERATIONS_PER_EPOCH=1
export ENET_OUTPUT_FOLDER="$nnUNet_results/${DATASET_NAME}/nnUNetTrainerENet_${RUN_NAME}__nnUNetPlans__2d/fold_0"
# ENET_BATCH_SIZE deliberately left unset -- first pass on this machine
# should use nnU-Net's own auto-planned batch size, purely to confirm the
# pipeline runs here at all before tuning anything for memory.

echo "=== Smoke test: ${RUN_NAME} ==="
echo "ENET_CHANNELS=${ENET_CHANNELS}"
echo "ENET_BOTTLENECKS=${ENET_BOTTLENECKS}"
echo "ENET_DECODER_TYPE=${ENET_DECODER_TYPE}"
echo "ENET_USE_DILATED=${ENET_USE_DILATED} ENET_USE_ASYMMETRIC=${ENET_USE_ASYMMETRIC} ENET_USE_STRIDED=${ENET_USE_STRIDED}"
echo "epochs=${ENET_EPOCHS} iters/epoch=${ENET_ITERATIONS_PER_EPOCH} val_iters/epoch=${ENET_VAL_ITERATIONS_PER_EPOCH}"
echo "output: ${ENET_OUTPUT_FOLDER}"

echo "=== Plan + preprocess (skipped if already done) ==="
if [ ! -f "$nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans.json" ]; then
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity
else
    echo "Plans already exist, skipping."
fi
if [ -f "$nnUNet_raw/${DATASET_NAME}/splits_final.json" ]; then
    cp "$nnUNet_raw/${DATASET_NAME}/splits_final.json" \
       "$nnUNet_preprocessed/${DATASET_NAME}/splits_final.json"
fi

cd "$REPO_ROOT/enet"

START_EPOCH_SECONDS=$(date +%s)
nnUNetv2_train "${DATASET_NAME}" 2d 0 -tr nnUNetTrainerENet
END_EPOCH_SECONDS=$(date +%s)

ELAPSED=$((END_EPOCH_SECONDS - START_EPOCH_SECONDS))
PER_EPOCH=$((ELAPSED / ENET_EPOCHS))
echo "=== Smoke test finished: ${RUN_NAME} ==="
echo "Elapsed: ${ELAPSED}s for ${ENET_EPOCHS} smoke epochs (~${ENET_ITERATIONS_PER_EPOCH} iters each, not full epochs)."
echo "This is NOT a real per-epoch time estimate (iteration count is truncated to ${ENET_ITERATIONS_PER_EPOCH})"
echo "-- use it only to confirm the pipeline runs end-to-end on this machine."
echo "For a real training-time estimate, time one UF run with ENET_ITERATIONS_PER_EPOCH unset"
echo "(full epoch) for a few epochs and extrapolate to 150."
