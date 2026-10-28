#!/bin/bash
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=finn_deployment_outputs/$(ls finn_deployment_outputs | grep S12_dense_512_u4_analytical_v1_ft15ep_preamble_ | tail -1)
echo "preamble: $PDIR"
nohup python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200 \
  --conv-order quantEnet_S12_dense_512_u4_analytical_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_dense_512_u4_analytical_v1_int6_fps250_lat200.json \
  --partitions all > /tmp/ooc_S12_512_full.log 2>&1 &
echo "launched $!"
disown -a
