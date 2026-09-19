#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_$(date +%Y%m%d_%H%M%S)
nohup bash -c "python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3.py finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_preamble_20260917_234018 /home/thelegendiv/finn/notebooks/enet/$OUTDIR && python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py /home/thelegendiv/finn/notebooks/enet/$OUTDIR" > /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_and_synth.log 2>&1 &
echo LAUNCHED_PID=$!
echo OUTDIR=$OUTDIR
