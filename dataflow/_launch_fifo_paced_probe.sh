#!/bin/bash
# Paced-source FIFO-sizing probe on partitions of an existing finn_auto build (see fifo_paced_probe.py).
# Usage (inside the FINN container): bash _launch_fifo_paced_probe.sh <preamble_dir> <tag> <target-fps> <paced|saturated> "<partition ids>" [max-workers=4]
#   e.g. bash _launch_fifo_paced_probe.sh finn_deployment_outputs/S12_dense_512_bilinear_int6_cap40_finnauto_v1_preamble_20261019_084330 S12_dense_512_bilinear_int6_cap40_finnauto_pacedprobe_v1 150 paced "1 2 6"
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT INITIAL_POOL_ROUTE
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
PDIR=$1; TAG=$2; FPS=$3; export FIFOSIM_SOURCE=$4; PARTS=$5; MAXW=${6:-4}; LOG=probe_${TAG}_p${PARTS// /_}.log
nohup python3 -u fifo_paced_probe.py $PDIR --tag $TAG --target-fps $FPS --fifo-autosize rtlsim --probe --partitions $PARTS --max-workers $MAXW > $LOG 2>&1 &
echo "launched $! (log $LOG)"
disown -a
