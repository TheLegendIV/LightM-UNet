#!/bin/bash
# Launch: v2 partition-6 build (reusing the identical-model v1 preamble) and the v2 whole-net preamble, in parallel.
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
nohup python3 -u rebuild_partition6_v2_and_rtlsim.py > /tmp/partition6_v2.log 2>&1 &
nohup python3 -u finn_s12_preamble.py quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep --tag S12_dense_256_u4_analytical_v2_ft15ep > /tmp/preamble_v2.log 2>&1 &
disown -a
echo launched
