#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 finn_ooc_12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512.py \
  finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600 \
  > /tmp/12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512.log 2>&1 &
disown
echo "launched, pid=$!"
