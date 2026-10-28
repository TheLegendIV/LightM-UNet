#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
STEP=force nohup bash run_arms.sh > /tmp/run_arms_full.log 2>&1 &
disown
echo "LAUNCHED run_arms.sh (STEP=force) PID=$!"
