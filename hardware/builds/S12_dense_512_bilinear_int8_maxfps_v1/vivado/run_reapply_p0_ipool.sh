#!/bin/bash
# Re-applies the partition 0 swap (InferPool initial-pool build) to the int8 BD project and forces a fresh synthesis state for it.
# Run with the project CLOSED in any Vivado GUI (a GUI session holding the old project overwrites these edits when it saves).
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
TMP=$FINN/notebooks/enet/finn_build_tmp
XPR=$FINN/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8/s12_512_bilin_int8.xpr
OLD_SUB=finn_build_tmp/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs
NEW=$TMP/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_ipool_p0
N=$FINN/notebooks/enet
for d in $XPR $NEW $N/swap_partition_repo.tcl $N/force_fresh_partition.tcl; do [ -e "$d" ] || { echo "MISSING $d"; exit 1; }; done
if ps -eo args | grep -i vivado | grep -v grep | grep -q "$XPR"; then echo "ABORT: a Vivado process has the project open"; exit 1; fi
cd /tmp
vivado -mode batch -nojournal -log /tmp/vivado_int8_swap_p0_b.log -source $N/swap_partition_repo.tcl -tclargs $XPR 0 $OLD_SUB $NEW > /dev/null 2>&1
grep -E "^(===|WARNING: [0-9]|ERROR: no stitched|VALIDATE_RESULT|GENERATE_RESULT|SWAP_DONE|UPGRADE_IP_ERROR)|^ERROR|Upgraded port|external port differences" /tmp/vivado_int8_swap_p0_b.log | cut -c1-300
echo "---- force fresh"
vivado -mode batch -nojournal -log /tmp/vivado_int8_fresh_p0.log -source $N/force_fresh_partition.tcl -tclargs $XPR 0 > /dev/null 2>&1
grep -E "^(===|top_|synth_1|impl_1|FRESH_DONE)|^ERROR|^CRITICAL" /tmp/vivado_int8_fresh_p0.log | cut -c1-250
echo "== xpr p0 repos"
grep -c 'ipool_p0' $XPR; grep -c '_pairs/GenericPartition_0' $XPR
echo SCRIPT_DONE
