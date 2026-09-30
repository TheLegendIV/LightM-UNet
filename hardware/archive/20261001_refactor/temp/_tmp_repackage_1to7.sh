#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
for i in 1 2 3 4 5 6 7; do
  echo "=== partition $i ==="
  vivado -mode batch -source /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/repackage_tcl/repackage_partition_${i}.tcl -nolog -nojournal 2>&1 | tail -8
done
