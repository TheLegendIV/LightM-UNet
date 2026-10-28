#!/bin/bash
# Full 8-way build (MILP folding bridge -> per-partition codegen/ipgen/FIFO/stitch -> OOC synth + rtlsim) of the uniform-INT8
# bilinear 512 network. Run inside the FINN container (docker exec finn_persistent bash <this>) after the preamble launch script.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=finn_deployment_outputs/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs_preamble_20261015_182343
echo "preamble: $PDIR"
nohup python3 -u finn_s12_build.py $PDIR \
  --tag S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs \
  --conv-order quantEnet_S12_dense_512_bilinear_int8_maxfps_v1_conv_order.json \
  --folding-json layer_bits_folding_S12_512_bilin_int8_maxfps_pairs.json \
  --partitions all > build_bilinear_int8_maxfps_full.log 2>&1 &
echo "launched $!"
disown -a
