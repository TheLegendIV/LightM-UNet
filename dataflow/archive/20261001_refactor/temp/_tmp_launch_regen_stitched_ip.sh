#!/bin/bash
set -e
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
source /tools/Xilinx/Vivado/2022.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh
cd /home/thelegendiv/finn/notebooks/enet
nohup python3 -u regen_stitched_ip_8way_full_v3.py > /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/regen_stitched_ip_8way_full_v3.log 2>&1 &
echo "launched PID $!"
