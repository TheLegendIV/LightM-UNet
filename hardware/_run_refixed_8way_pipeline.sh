#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet

python3 finn_hawq_preamble_12_dense_relu_warmstart150ep_alpha025_trained.py 2>&1 | tee /tmp/preamble_refixed.log

LATEST=$(ls -dt finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_* | head -1)
echo "LATEST_PREAMBLE_DIR=$LATEST"

OUTDIR="finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_$(date +%Y%m%d_%H%M%S)_refixed"
echo "OUTDIR=$OUTDIR"

python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py "$LATEST" "/home/thelegendiv/finn/notebooks/enet/$OUTDIR"
python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py "/home/thelegendiv/finn/notebooks/enet/$OUTDIR"
