#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh
PREAMBLE=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_$(date +%Y%m%d_%H%M%S)
echo "OUTDIR=$OUTDIR"
nohup python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram.py \
    "$PREAMBLE" "/home/thelegendiv/finn/notebooks/enet/$OUTDIR" \
    > "/tmp/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "PID=$!"
disown
sleep 2
ps aux | grep '[f]inn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram'
