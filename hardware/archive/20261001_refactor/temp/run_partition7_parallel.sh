#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v2_20260918
LOG=/tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_partition7_parallel.log
nohup python3 /tmp/run_single_partition_synth.py /home/thelegendiv/finn/notebooks/enet/$OUTDIR 7 > "$LOG" 2>&1 &
echo "LAUNCHED PID=$!"
