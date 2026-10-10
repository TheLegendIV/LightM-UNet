#!/bin/bash
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
OUT=finn_deployment_outputs/S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261010_050602
nohup python3 -u finn_add_iodma_and_driver.py $OUT \
  --original-build-dir finn_build_tmp/S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200 \
  > /tmp/iodma_512.log 2>&1 &
echo "launched $!"
disown -a
