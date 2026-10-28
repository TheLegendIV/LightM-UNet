#!/bin/bash
# Fresh-synthesis reset of partition 0 in the int8 BD project (project must be closed in any GUI).
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
XPR=$FINN/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8/s12_512_bilin_int8.xpr
N=$FINN/notebooks/enet
if ps -eo args | grep -i vivado | grep -v grep | grep -q "$XPR"; then echo "ABORT: a Vivado process has the project open"; exit 1; fi
cd /tmp
vivado -mode batch -nojournal -log /tmp/vivado_int8_fresh_p0.log -source $N/force_fresh_partition.tcl -tclargs $XPR 0 > /dev/null 2>&1
grep -E "^(===|top_|synth_1|impl_1|[0-9]+$|FRESH_DONE)|^ERROR|^CRITICAL" /tmp/vivado_int8_fresh_p0.log | cut -c1-250
echo "== xpr p0 repos (ipool_p0 / old)"
grep -c 'ipool_p0' $XPR; grep -c '_pairs/GenericPartition_0' $XPR
echo SCRIPT_DONE
