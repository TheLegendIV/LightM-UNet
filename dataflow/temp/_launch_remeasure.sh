#!/bin/bash
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
unset VERILATOR_ROOT
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1
cd /home/thelegendiv/finn/notebooks/enet
O=finn_deployment_outputs
i=0
for d in \
  $O/ratchet_ablation_finn_autofold_autofold_partition2_20261009_154752 \
  $O/ratchet_1pct_simfifo_milpfold_partition2_20261009_154757 \
  $O/ratchet_25pct_simfifo_milpfold_partition2_20261009_154802 \
  $O/ratchet_100pct_simfifo_milpfold_partition2_20261009_155407 \
  $O/ratchet_200pct_simfifo_milpfold_partition2_20261009_175218 \
  $O/ratchet_off_simfifo_milpfold_partition2_20261009_180323; do
  i=$((i+1))
  nohup python3 -u _remeasure_rtlsim.py "$d" > /tmp/remeasure_rtlsim_$i.log 2>&1 &
  echo "launched $i $d PID=$!"
done
