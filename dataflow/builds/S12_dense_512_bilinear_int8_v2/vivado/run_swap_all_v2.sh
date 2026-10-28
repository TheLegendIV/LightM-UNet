#!/bin/bash
# Runs inside finn_persistent. Re-points the int8 BD project (S12_512_bilin_int8) at the v2 build (all 8 partitions from finn_build_tmp/S12_dense_512_bilinear_int8_v2) and resets its synthesis state.
# Backs up the xpr + BD sources first (backup dir _backup_pre_v2_20261017). Run with the project CLOSED in every Vivado GUI (a GUI session holding the old project overwrites these edits when it saves).
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
TMP=$FINN/notebooks/enet/finn_build_tmp
PROJ=$FINN/vivado_projects/S12_512_bilin_int8/s12_512_bilin_int8
XPR=$PROJ/s12_512_bilin_int8.xpr
NEW=$TMP/S12_dense_512_bilinear_int8_v2
TCL=$FINN/notebooks/enet/swap_all_partitions_repo.tcl
for d in $XPR $NEW $TCL; do [ -e "$d" ] || { echo "MISSING $d"; exit 1; }; done
if ps -eo args | grep -i vivado | grep -v grep | grep -q "s12_512_bilin_int8"; then echo "ABORT: a Vivado process has the project open"; exit 1; fi
BK=$FINN/vivado_projects/S12_512_bilin_int8/_backup_pre_v2_20261017
if [ -e "$BK" ]; then echo "backup exists, reusing: $BK"; else mkdir -p $BK && cp -r $XPR $PROJ/s12_512_bilin_int8.srcs $BK/; fi
# The IODMA HLS IPs of partitions 0 / 7 live under an old build dir; take their repo paths from the backup xpr
IODMA=$(grep -o 'finn_build_tmp/[A-Za-z0-9_]*/GenericPartition_[0-7]/[^"]*IODMA[^"]*' $BK/s12_512_bilin_int8.xpr | sort -u | sed "s#^#$FINN/notebooks/enet/#")
echo "IODMA repos: $IODMA"
for r in $IODMA; do [ -e "$r" ] || { echo "MISSING IODMA repo $r"; exit 1; }; done
cd /tmp
vivado -mode batch -nojournal -log /tmp/vivado_int8_swap_all.log -source $TCL -tclargs $XPR $NEW $IODMA > /dev/null 2>&1
grep -E "^(===|partition [0-7]:|WARNING: [0-9]|ERROR: no stitched|VALIDATE_RESULT|GENERATE_RESULT|SWAP_ALL_DONE|UPGRADE_IP_ERROR|RUN |old-pool)|^(ERROR|CRITICAL WARNING)|Upgraded port|external port differences" /tmp/vivado_int8_swap_all.log | cut -c1-300
echo "== build dirs referenced by the xpr"
grep -o "finn_build_tmp/[A-Za-z0-9_]*/GenericPartition_[0-7]" $XPR | sort | uniq -c
echo SCRIPT_DONE
