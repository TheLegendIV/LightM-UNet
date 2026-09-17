#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
mkdir -p /home/thelegendiv/finn/finn_build_persist
export FINN_BUILD_DIR=/home/thelegendiv/finn/finn_build_persist
PREAMBLE_DIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_preamble_20260917_005512
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_$(date +%Y%m%d_%H%M%S)
nohup bash -c "python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full.py $PREAMBLE_DIR /home/thelegendiv/finn/notebooks/enet/$OUTDIR && python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py /home/thelegendiv/finn/notebooks/enet/$OUTDIR" > /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_and_synth.log 2>&1 &
disown
echo "launched pid $!"
echo "OUTDIR=$OUTDIR"
