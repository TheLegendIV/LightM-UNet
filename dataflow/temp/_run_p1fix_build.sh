#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
nohup python3 finn_s12_build.py finn_deployment_outputs/u8in_signbias2_preamble_20261008_174934 \
  --tag S12_dense_256_u4_u8in_int6_p1fix \
  --conv-order /tmp/conv_order.json \
  --folding-json /tmp/layer_bits_folding_final.json \
  --probe --partitions 1 > /tmp/p1fix_build.log 2>&1 &
echo $! > /tmp/p1fix_build.pid
