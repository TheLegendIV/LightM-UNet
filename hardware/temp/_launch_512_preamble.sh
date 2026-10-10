#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh
nohup python3 -u finn_s12_preamble.py quantEnet_S12_dense_512_u4_analytical_v1_ft15ep_u8in --tag S12_dense_512_u4_analytical_v1_ft15ep > /tmp/preamble_512.log 2>&1 &
disown -a
echo launched
