#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd /tmp
for i in 1 2 3 4 5 6 7; do
  vivado -mode batch -source /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/repackage_tcl/repackage_partition_${i}.tcl -nolog -nojournal > /tmp/repack${i}_v2.log 2>&1
done
echo ALL_DONE
