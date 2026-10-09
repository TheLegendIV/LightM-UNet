#!/bin/bash
# Launch the S12 dense 256 ratchet ablation (5 simfifo arms + FINN control) with the gate forced.
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
mkdir -p "$FINN_BUILD_DIR"
cd /home/thelegendiv/finn/notebooks/enet
STEP=force PARTS=2 nohup bash run_arms.sh > /tmp/ratchet256_run_p2_force.log 2>&1 &
echo "launched run_arms.sh PID=$!"
