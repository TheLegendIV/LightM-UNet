#!/bin/bash
# Rebuild only partition 7 (production pipeline, --probe) of a finished/in-progress job with the FIFO-landing fix; output goes to a new dir.
# Usage: bash _rebuild_p7.sh <stem> <job-tag> <folding json> <preamble-dir-name>
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
NET=$1; TAG=$2; FOLD=$3; PDIR=finn_deployment_outputs/$4
nohup python3 -u finn_s12_build.py $PDIR --tag ${TAG}_p7fix --conv-order ${NET}_conv_order.json --folding-json $FOLD --partitions 7 --probe > build_${TAG}_p7fix.log 2>&1 &
echo "launched $!"
disown -a
