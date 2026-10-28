#!/bin/bash
# Preamble (+ bridge/FIFO landing gates, no HLS/Vivado build) for the uniform-INT8 bilinear 512 network folded from
# MILP/artifacts/S12_512_bilin_int8_maxfps/simfifo_pairs. Run inside the FINN container (docker exec finn_persistent bash <this>).
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
nohup python3 -u finn_s12_preamble.py quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_u8in \
  --tag S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs \
  --conv-order quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json \
  > preamble_bilinear_int8_maxfps.log 2>&1 &
disown -a
echo launched
