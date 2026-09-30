#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_$(date +%Y%m%d_%H%M%S)
nohup bash -c "python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260908_211732 /home/thelegendiv/finn/notebooks/enet/$OUTDIR && python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py /home/thelegendiv/finn/notebooks/enet/$OUTDIR" > /tmp/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_and_synth.log 2>&1 &
echo "launched PID $! OUTDIR=$OUTDIR"
