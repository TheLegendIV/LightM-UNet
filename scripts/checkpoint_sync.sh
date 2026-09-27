#!/bin/bash
# One-click sync of nnU-Net checkpoints from the HPC cluster down to this
# repo's local data/nnUNet_results/, so compression/collect_results.py can
# be run locally against freshly-finished HPC training runs. Uses rsync, so
# re-running this only transfers what actually changed -- already-synced
# checkpoints are skipped, not re-downloaded.
#
# Usage:
#   scripts/checkpoint_sync.sh                                    # sync ALL of nnUNet_results
#   scripts/checkpoint_sync.sh Dataset509_ARCADE_1x1_4c           # one dataset
#   scripts/checkpoint_sync.sh Dataset509_ARCADE_1x1_4c/nnUNetTrainerENet_foo__nnUNetPlans__2d   # one run
#   HPC_USER=someoneelse scripts/checkpoint_sync.sh               # override the default user
#
# Config (env vars, all overridable):
#   HPC_USER  -- your HPC username (default: yhussein, per the paths logged
#                in this repo's own debug.json files -- /home/yhussein/...)
#   HPC_HOST  -- SSH login node (default: snellius.surf.nl -- NOT a compute
#                node hostname like gcn2.local.snellius.surf.nl, those are
#                only reachable from inside a running Slurm job)
#   HPC_ROOT  -- repo path on HPC, relative to $HOME (default: LightM-UNet)
#
# Skips validation/ (collect_results.py runs its own separate inference on
# imagesTs/labelsTs -- it never reads the internal CV val-split PNGs under
# validation/) and labelsPr_*/ (prediction caches, can be huge, regenerable
# any time from a checkpoint) to avoid syncing bytes nothing actually needs.

set -euo pipefail

HPC_USER="${HPC_USER:-yhussein}"
HPC_HOST="${HPC_HOST:-snellius.surf.nl}"
HPC_ROOT="${HPC_ROOT:-LightM-UNet}"

SUBPATH="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_BASE="${HPC_USER}@${HPC_HOST}:${HPC_ROOT}/data/nnUNet_results/${SUBPATH}"
LOCAL_BASE="${REPO_ROOT}/data/nnUNet_results/${SUBPATH}"
# Strip any trailing slash then add exactly one, so rsync always syncs
# CONTENTS of the source dir into the dest dir (consistent whether SUBPATH
# is empty, a dataset name, or a full run path).
REMOTE="${REMOTE_BASE%/}/"
LOCAL="${LOCAL_BASE%/}/"

mkdir -p "$LOCAL"

echo "=== Syncing checkpoints ==="
echo "From: ${REMOTE}"
echo "To:   ${LOCAL}"
echo

rsync -avzP \
    --exclude='validation/' \
    --exclude='labelsPr_*/' \
    "$REMOTE" "$LOCAL"

echo
echo "Done. Run compression/collect_results.py for any newly-synced runs to update results.csv."
