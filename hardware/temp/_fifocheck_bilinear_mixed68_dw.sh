#!/bin/bash
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
python3 -u finn_s12_build.py finn_deployment_outputs/S12_dense_512_bilinear_mixed68_v1_ft15ep_dw_preamble_20261013_000943 \
  --tag S12_dense_512_bilinear_mixed68_v1_ft15ep_dw \
  --conv-order quantEnet_S12_dense_512_bilinear_mixed68_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_dense_512_bilinear_mixed68_v1_dw_simfifo.json \
  --partitions all --fifo-check > fifocheck_bilinear_mixed68_dw.log 2>&1
echo EXIT=$?
