#!/bin/bash
# relaunch the bilinear preamble with faulthandler, dump traceback via SIGABRT after $1 seconds
pkill -f finn_s12_preamble.py
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
export PYTHONFAULTHANDLER=1
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh
python3 -u finn_s12_preamble.py quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in --tag S12_dense_256_u4_bilinear_analytical_v1_ft15ep > /tmp/preamble_bilinear2.log 2>&1 &
PID=$!
sleep ${1:-90}
if kill -0 $PID 2>/dev/null; then
  kill -ABRT $PID
  sleep 3
  echo "dumped"
else
  echo "finished before dump"
fi
grep -n -A40 "Fatal Python error" /tmp/preamble_bilinear2.log | cut -c1-200 | head -70
