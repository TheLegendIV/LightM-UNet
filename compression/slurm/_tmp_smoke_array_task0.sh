#!/bin/bash
# Local Docker smoke test of qat_12_separable_dense_relu_joint_alpha_sweep_array.job's
# TASK_ID=0 (alpha=0.0) pipeline logic -- mirrors the job file's STEP 1/STEP 2/collect
# steps exactly, adapted for the Docker container (no SLURM/module/conda-activate,
# /workspace/LightM-UNet instead of $HOME/LightM-UNet), with QAT_EPOCHS=1 (not the
# job file's real default of 5) purely to keep this smoke test fast -- the mechanics
# (env vars, resume logic, self-healing check, collect_results.py row) are what's
# being verified here, not model quality.
set -e

DATASET_ID=509
DATASET_NAME="Dataset509_ARCADE_1x1_4c"
TRAINER_CLASS="nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock"
FP32_SOURCE_NET_NAME="nnUNetTrainerENet_12_separable_dense_relu"
QAT_EPOCHS=1

BLOCK_BITS_FILE="compression/hawq/artifacts/block_bits_folding_12_separable_dense_relu_joint_alpha0.0.json"
CALIBRATED_NET_NAME="nnUNetTrainerCombinedQuantENet_12_separable_dense_relu_perblock_joint_alpha0.0_calibrated"
RUN_NAME="12_separable_dense_relu_joint_alpha0.0_smoketest${QAT_EPOCHS}ep"

export nnUNet_raw="/workspace/LightM-UNet/data/nnUNet_raw"
export nnUNet_preprocessed="/workspace/LightM-UNet/data/nnUNet_preprocessed"
export nnUNet_results="/workspace/LightM-UNet/data/nnUNet_results"
export PYTHONHASHSEED=42

cd /workspace/LightM-UNet

FP32_CHECKPOINT="$nnUNet_results/${DATASET_NAME}/${FP32_SOURCE_NET_NAME}__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
echo "=== Checking FP32 source checkpoint exists ==="
if [ ! -f "$FP32_CHECKPOINT" ]; then
    echo "ERROR: FP32 source checkpoint not found: $FP32_CHECKPOINT" >&2
    exit 1
fi
echo "OK: $FP32_CHECKPOINT"

CALIBRATED_CHECKPOINT="$nnUNet_results/${DATASET_NAME}/${CALIBRATED_NET_NAME}__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
echo "=== STEP 1: calibrate ==="
if [ -f "$CALIBRATED_CHECKPOINT" ]; then
    echo "Calibrated checkpoint already exists -- skipping."
else
    python compression/post-quantization/calibrate_12_separable_dense_relu_perblock.py \
        --source-net-name "${FP32_SOURCE_NET_NAME}" \
        --out-net-name "${CALIBRATED_NET_NAME}" \
        --dataset-name "${DATASET_NAME}" \
        --block-bits-file "${BLOCK_BITS_FILE}"
fi

echo "=== STEP 2: real QAT fine-tuning, ${QAT_EPOCHS}-epoch smoke schedule ==="
export ENET_BLOCK_BITS_FILE="/workspace/LightM-UNet/${BLOCK_BITS_FILE}"
export ENET_PRETRAINED_CHECKPOINT="${CALIBRATED_CHECKPOINT}"
export ENET_EPOCHS=${QAT_EPOCHS}
export ENET_OUTPUT_FOLDER="$nnUNet_results/${DATASET_NAME}/${TRAINER_CLASS}_${RUN_NAME}__nnUNetPlans__2d/fold_0"
# Smoke-test-only speed cap -- real sweep runs use nnU-Net's real default
# (250 train / 50 val iterations per epoch); this just proves the pipeline
# mechanics (env vars, resume logic, self-healing check, results.csv row)
# work, not model quality, so cut iterations way down.
export ENET_ITERATIONS_PER_EPOCH=10
export ENET_VAL_ITERATIONS_PER_EPOCH=2

cd /workspace/LightM-UNet/enet

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

echo "=== Self-healing check ==="
if [ -f "${ENET_OUTPUT_FOLDER}/checkpoint_final.pth" ] && [ -f "${ENET_OUTPUT_FOLDER}/checkpoint_latest.pth" ]; then
    FINAL_EPOCH_CHECK=$(python -c "import torch; print(torch.load('${ENET_OUTPUT_FOLDER}/checkpoint_final.pth', map_location='cpu', weights_only=False)['current_epoch'])")
    LATEST_EPOCH_CHECK=$(python -c "import torch; print(torch.load('${ENET_OUTPUT_FOLDER}/checkpoint_latest.pth', map_location='cpu', weights_only=False)['current_epoch'])")
    if [ "$LATEST_EPOCH_CHECK" -gt "$FINAL_EPOCH_CHECK" ]; then
        echo "Promoting stale checkpoint_final.pth."
        cp "${ENET_OUTPUT_FOLDER}/checkpoint_final.pth" "${ENET_OUTPUT_FOLDER}/checkpoint_final_STALE_epoch${FINAL_EPOCH_CHECK}_backup.pth"
        cp "${ENET_OUTPUT_FOLDER}/checkpoint_latest.pth" "${ENET_OUTPUT_FOLDER}/checkpoint_final.pth"
    fi
fi

echo "=== Checking training progress ==="
EXISTING_CHECKPOINT=""
if [ -f "${ENET_OUTPUT_FOLDER}/checkpoint_final.pth" ]; then
    EXISTING_CHECKPOINT="${ENET_OUTPUT_FOLDER}/checkpoint_final.pth"
elif [ -f "${ENET_OUTPUT_FOLDER}/checkpoint_latest.pth" ]; then
    EXISTING_CHECKPOINT="${ENET_OUTPUT_FOLDER}/checkpoint_latest.pth"
fi

if [ -z "$EXISTING_CHECKPOINT" ]; then
    echo "=== Start training (fresh): ${RUN_NAME} ==="
    nnUNetv2_train "${DATASET_NAME}" 2d 0 -tr "${TRAINER_CLASS}"
else
    CURRENT_EPOCH=$(cd /workspace/LightM-UNet && python -c "import torch; print(torch.load('${EXISTING_CHECKPOINT}', map_location='cpu', weights_only=False)['current_epoch'])")
    echo "Existing checkpoint at epoch ${CURRENT_EPOCH}/${ENET_EPOCHS}."
    if [ "$CURRENT_EPOCH" -ge "$ENET_EPOCHS" ]; then
        echo "Schedule already complete -- skipping training."
    else
        echo "=== Resuming with --c ==="
        nnUNetv2_train "${DATASET_NAME}" 2d 0 -tr "${TRAINER_CLASS}" --c
    fi
fi
echo "Training step finished for ${RUN_NAME}."

echo "=== Collecting results ==="
cd /workspace/LightM-UNet
FINAL_EPOCH=$(python -c "import torch; print(torch.load('${ENET_OUTPUT_FOLDER}/checkpoint_final.pth', map_location='cpu', weights_only=False)['current_epoch'])" 2>/dev/null || echo -1)
if [ "$FINAL_EPOCH" -ge "$ENET_EPOCHS" ]; then
    python compression/collect_results.py \
        --net-name "${TRAINER_CLASS}_${RUN_NAME}" \
        --stage "${RUN_NAME}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-id "${DATASET_ID}" \
        --channels 4,16,32,16,4 \
        --bottlenecks 4,8,8,2,1 \
        --decoder-type upsample_conv \
        --use-asymmetric 0 \
        --context-pattern dense_dilation \
        --separable-dilated 1 \
        --quant-bits 8 \
        --trainer-class "${TRAINER_CLASS}"
else
    echo "Only at epoch ${FINAL_EPOCH}/${ENET_EPOCHS} -- skipping collect_results.py."
fi

echo "SMOKE TEST DONE: ${RUN_NAME}."
