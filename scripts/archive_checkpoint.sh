#!/usr/bin/env bash
# Archive a finished training run's checkpoints + logs out of the repo's
# nnUNet_results tree (never pushed, always regenerable/disposable on HPC)
# into a permanent, timestamped home at ~/models/. Call this once a training
# job's nnUNetv2_train call has returned -- see train_enetpost.job /
# train_enetpostrefinement.job for the call site.
#
# Usage:
#   bash scripts/archive_checkpoint.sh <DATASET_NAME> <TRAINER_NAME> [CONFIGURATION] [FOLD] [PLANS_NAME]
#
# Example:
#   bash scripts/archive_checkpoint.sh Dataset501_ARCADE nnUNetTrainerENet 2d 0 nnUNetPlans
#
# Requires $nnUNet_results to already be exported (same variable the training
# job itself uses -- see the "-- nnU-Net paths --" section of the .job files).
set -euo pipefail

DATASET_NAME="${1:?Usage: archive_checkpoint.sh <DATASET_NAME> <TRAINER_NAME> [CONFIGURATION] [FOLD] [PLANS_NAME]}"
TRAINER_NAME="${2:?Usage: archive_checkpoint.sh <DATASET_NAME> <TRAINER_NAME> [CONFIGURATION] [FOLD] [PLANS_NAME]}"
CONFIGURATION="${3:-2d}"
FOLD="${4:-0}"
PLANS_NAME="${5:-nnUNetPlans}"

: "${nnUNet_results:?nnUNet_results is not set -- source the training job's env exports first}"
: "${MODELS_ARCHIVE_DIR:=$HOME/models}"

SRC_DIR="$nnUNet_results/${DATASET_NAME}/${TRAINER_NAME}__${PLANS_NAME}__${CONFIGURATION}/fold_${FOLD}"
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: training output not found at $SRC_DIR" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEST_DIR="${MODELS_ARCHIVE_DIR}/${TRAINER_NAME}_${DATASET_NAME}_${TIMESTAMP}"
mkdir -p "$DEST_DIR"

echo "=== Archiving checkpoint ==="
echo "Source : $SRC_DIR"
echo "Dest   : $DEST_DIR"

# Checkpoints -- copy whichever of these nnU-Net actually wrote (best always
# should exist; latest/final depend on whether training completed normally
# vs. was interrupted and resumed).
for ckpt in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
    if [ -f "$SRC_DIR/$ckpt" ]; then
        cp "$SRC_DIR/$ckpt" "$DEST_DIR/"
        echo "  copied $ckpt"
    fi
done

# Small metadata -- cheap, useful for later comparison/reproducibility.
# Deliberately NOT copied: the validation/ subfolder (per-case prediction
# PNGs from final validation) -- large, disposable, regenerable from the
# archived checkpoint via nnUNetv2_predict if ever needed again.
for f in progress.png debug.json plans.json dataset.json dataset_fingerprint.json; do
    if [ -f "$SRC_DIR/$f" ]; then
        cp "$SRC_DIR/$f" "$DEST_DIR/"
    fi
done

# Training logs -- variable date suffix in the filename, so glob directly.
for f in "$SRC_DIR"/training_log_*.txt; do
    [ -f "$f" ] && cp "$f" "$DEST_DIR/"
done

echo "=== Archived files ==="
ls -la "$DEST_DIR"
echo "=== Total size ==="
du -sh "$DEST_DIR"
echo "Done."
