#!/bin/bash
# Kicks off the clDice-weight Pareto sweep for ENetComboEG1: one independent,
# self-continuing SLURM job chain per weight (see train_combo_eg1_cldice_pareto.job
# for why each is a chain rather than a single job -- 150 epochs won't fit in
# the cluster's 6h time limit in one slot on a MIG GPU).
#
# Usage: bash submit_combo_eg1_cldice_pareto.sh
set -e

cd "$(dirname "$0")"

for w in 0.0 0.25 0.5 1.0 2.0; do
    echo "Submitting COMBO_CLDICE_WEIGHT=${w}..."
    sbatch --export=ALL,COMBO_CLDICE_WEIGHT="$w",RESUBMIT_COUNT=0 train_combo_eg1_cldice_pareto.job
done

echo "All 5 chains submitted. Check progress with: squeue -u \$USER"
