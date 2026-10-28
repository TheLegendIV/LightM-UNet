#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918
PREAMBLE_DIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_preamble_20260917_234018
LOG=/tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_and_synth_v2.log
nohup bash -c "python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full.py $PREAMBLE_DIR /home/thelegendiv/finn/notebooks/enet/$OUTDIR && python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py /home/thelegendiv/finn/notebooks/enet/$OUTDIR" > "$LOG" 2>&1 &
echo "LAUNCHED PID=$!"
