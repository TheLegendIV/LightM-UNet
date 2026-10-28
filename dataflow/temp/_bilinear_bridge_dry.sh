#!/bin/bash
# Bridge dry run (folding + FIFO plan, no HLS / Vivado) of the bilinear net against the landed preamble.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=finn_deployment_outputs/$(ls finn_deployment_outputs | grep S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_ | tail -1)
python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200 \
  --conv-order quantEnet_S12_dense_256_u4_bilinear_analytical_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_dense_256_u4_bilinear_analytical_v1_int6_fps250_lat200.json \
  --partitions all --bridge-only > /tmp/bilinear_bridge.log 2>&1
echo "exit=$?"
grep -vE "Warning|warn" /tmp/bilinear_bridge.log | tail -40 | cut -c1-220
grep -cE "MISMATCH|NO MATCH" /tmp/bilinear_bridge.log
