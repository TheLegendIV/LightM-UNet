#!/bin/bash
# Runs inside finn_persistent. Swaps partition 0 of the int8 BD project to the InferPool-initial-pool build (ft15ep_ipool_p0).
# Backs the project up first (xpr + BD sources only; no run results exist in it).
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
TMP=$FINN/notebooks/enet/finn_build_tmp
PROJ=$FINN/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8
XPR=$PROJ/s12_512_bilin_int8.xpr
OLD_SUB=finn_build_tmp/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs
NEW=$TMP/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_ipool_p0
TCL=$FINN/notebooks/enet/swap_partition_repo.tcl
for d in $XPR $NEW $TCL; do [ -e "$d" ] || { echo "MISSING $d"; exit 1; }; done
BK=$FINN/vivado_projects/S12_512_bilin_int8/_backup_pre_ipool_p0
[ -e "$BK" ] && { echo "ABORT: $BK exists"; exit 1; }
mkdir -p $BK && cp -r $PROJ/s12_512_bilin_int8.xpr $PROJ/s12_512_bilin_int8.srcs $BK/
cd /tmp
vivado -mode batch -nojournal -log /tmp/vivado_int8_swap_p0.log -source $TCL -tclargs $XPR 0 $OLD_SUB $NEW > /dev/null 2>&1
grep -E "^(===|WARNING: [0-9]|ERROR: no stitched|VALIDATE_RESULT|GENERATE_RESULT|SWAP_DONE|UPGRADE_IP_ERROR)|^(ERROR|CRITICAL WARNING)|Upgraded port|external port differences" /tmp/vivado_int8_swap_p0.log | cut -c1-300
echo "== partition 0 repos in the xpr"
grep -o "finn_build_tmp/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_[a-z0-9_]*/GenericPartition_0" $XPR | sort | uniq -c
echo SCRIPT_DONE
