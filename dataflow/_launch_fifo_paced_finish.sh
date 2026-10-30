#!/bin/bash
# Finish (split FIFOs, stitch, OOC synth, rtlsim) the paced-FIFO probe partitions. See fifo_paced_finish.py.
# Usage (container): bash _launch_fifo_paced_finish.sh "<ids>" [max-workers=4]
export HOME=/tmp/home_dir
unset VERILATOR_ROOT INITIAL_POOL_ROUTE
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
T=S12_dense_512_bilinear_int6_cap40_finnauto_pacedprobe_v1
P=finn_deployment_outputs/S12_dense_512_bilinear_int6_cap40_finnauto_v1_preamble_20261019_084330
nohup python3 -u fifo_paced_finish.py $P $T finn_deployment_outputs/${T}_finish $1 --max-workers ${2:-4} > finish_$T.log 2>&1 &
echo "launched $!"
disown -a
