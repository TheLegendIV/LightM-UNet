#!/bin/bash
# Build the tap testbench for partition 1 in the background (verilate + compile).
export HOME=/tmp/home_dir
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 rtlsim_taps.py --build-tmp finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200 --partition 1 > /tmp/taps_build_p1.log 2>&1 &
echo started $!
