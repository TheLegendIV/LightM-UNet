#!/bin/bash
# Rebuild of partitions 1, 5 and 7 of the uniform-INT8 bilinear 512 network with the fixed bridge:
#   - name-stable FIFO insertion (p1 deadlock / p5 mis-sized up4 FIFOs came from renumbered node names)
#   - LabelSelect PE = labels, final bias PE, final FMPadding_Pixel SIMD (p7)
# Production pipeline on the subset (--probe): prefix names, stitched IP, OOC synth, per-partition rtlsim; no combine step.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=finn_deployment_outputs/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs_preamble_20261015_182343
nohup python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_p157fix \
  --conv-order quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json \
  --partitions 1 5 7 --probe > build_p157_fix.log 2>&1 &
echo "launched $!"
disown -a
