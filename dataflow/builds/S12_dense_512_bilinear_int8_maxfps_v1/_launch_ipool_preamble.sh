#!/bin/bash
# Preamble variant: the initial block's 1-channel pool branch lowered with InferPool (depthwise SWG + Pool_hls) instead of the unpipelined
# StreamingMaxPool_hls. Experiment for partition 0 (Vivado OOC crash on the InferPool route was seen in bottleneck_probe_v1).
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
export INITIAL_POOL_ROUTE=infer_pool
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
nohup python3 -u finn_s12_preamble.py quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_u8in \
  --tag S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_ipool \
  --conv-order quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json \
  > preamble_bilinear_int8_ipool.log 2>&1 &
disown -a
echo launched
