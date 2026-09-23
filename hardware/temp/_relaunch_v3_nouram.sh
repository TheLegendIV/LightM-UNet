#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_$(date +%Y%m%d_%H%M%S)
nohup python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram.py \
  finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600 \
  /home/thelegendiv/finn/notebooks/enet/$OUTDIR \
  > /tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_relaunch.log 2>&1 &
echo "LAUNCHED_PID=$!"
echo "$OUTDIR" > /tmp/last_relaunch_outdir.txt
sleep 2
cat /tmp/last_relaunch_outdir.txt
