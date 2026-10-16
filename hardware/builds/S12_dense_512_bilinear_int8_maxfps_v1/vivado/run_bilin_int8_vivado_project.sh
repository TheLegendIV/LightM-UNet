#!/bin/bash
# Runs inside finn_persistent. Duplicates the hand-edited mixed68 BD project (clocking, latency monitor, IODMA IPs kept) as
# s12_512_bilin_int8 and points the GenericPartition IP repos at the INT8 build: partitions 0,2,3,4,6 from the full 8-way build,
# 1,5,7 from the rebuild with the name-stable FIFO fix and LabelSelect PE 5.
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
TMP=$FINN/notebooks/enet/finn_build_tmp
BASE_XPR=$FINN/vivado_projects/S12_512_bilin_mixed68/s12_512_bilin_mixed68/s12_512_bilin_mixed68.xpr
OLD_SUB=finn_build_tmp/S12_dense_512_bilinear_mixed68_v1_ft15ep_dw
MAIN=$TMP/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_pairs
ALT=$TMP/S12_dense_512_bilinear_int8_maxfps_v1_ft15ep_p157fix
PROJ_NAME=s12_512_bilin_int8
PROJ_DIR=$FINN/vivado_projects/S12_512_bilin_int8/$PROJ_NAME
TCL=$FINN/notebooks/enet/make_project_from_base.tcl
for d in $BASE_XPR $MAIN $ALT $TCL; do [ -e "$d" ] || { echo "MISSING $d"; exit 1; }; done
[ -e "$PROJ_DIR" ] && { echo "ABORT: $PROJ_DIR exists"; exit 1; }
mkdir -p $(dirname $PROJ_DIR)
cd /tmp
vivado -mode batch -nojournal -log /tmp/vivado_int8_make.log -source $TCL \
  -tclargs $BASE_XPR $PROJ_NAME $PROJ_DIR $OLD_SUB $MAIN $ALT "1 5 7" > /dev/null 2>&1
grep -E "^(===|partition [0-7]:|WARNING: partition|ERROR: no stitched|VALIDATE_RESULT|GENERATE_RESULT|MAKE_PROJECT_DONE|UPGRADE_IP_ERROR)|^(ERROR|CRITICAL WARNING)" /tmp/vivado_int8_make.log | cut -c1-400
P=$PROJ_DIR
echo "== build dirs referenced by the new xpr"
grep -o "finn_build_tmp/[A-Za-z0-9_]*" $P/$PROJ_NAME.xpr | sort | uniq -c
echo SCRIPT_DONE
