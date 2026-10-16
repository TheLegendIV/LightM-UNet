#!/bin/bash
# Partition 0 only (production pipeline: prefix names, stitched IP, OOC synth, rtlsim) with the InferPool-lowered initial pool.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=finn_deployment_outputs/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_ipool_preamble_20261016_105235
nohup python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_ipool_p0 \
  --conv-order quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json \
  --partitions 0 --probe > build_ipool_p0.log 2>&1 &
echo "launched $!"
disown -a
