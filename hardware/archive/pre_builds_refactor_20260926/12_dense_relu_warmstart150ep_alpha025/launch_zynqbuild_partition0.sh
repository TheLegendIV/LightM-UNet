#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
PREAMBLE_DIR=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_204947
OUTDIR=finn_deployment_outputs/zynqbuild_12_dense_relu_warmstart150ep_alpha025_trained_partition0_$(date +%Y%m%d_%H%M%S)
nohup python3 finn_zynqbuild_12_dense_relu_alpha025_partition0.py "$PREAMBLE_DIR" "/home/thelegendiv/finn/notebooks/enet/$OUTDIR" \
    > /tmp/zynqbuild_12_dense_relu_alpha025_partition0.log 2>&1 &
echo "LAUNCHED_PID=$!"
echo "OUTDIR=$OUTDIR"
