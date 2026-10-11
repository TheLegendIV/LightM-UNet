#!/bin/bash
# Runs inside finn_persistent. Copy of S12_256_analytical whose FINN partition IPs come from the
# bilinear 256 build. The bilinear build has no IODMA IPs, so the already-repackaged
# p0_input/p7_output IODMA IPs from the nearest-neighbour 256 build are kept (same I/O geometry).
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
BUILD=$FINN/notebooks/enet/finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200
OLD=$FINN/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200
IODMA_P0=$OLD/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip
IODMA_P7=$OLD/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip
BASE_XPR=$FINN/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
PROJ_NAME=s12_256_bilinear_analytical
PROJ_DIR=$FINN/vivado_projects/S12_256_bilinear_analytical/$PROJ_NAME
cd /tmp
for d in $BUILD $IODMA_P0/component.xml $IODMA_P7/component.xml $BASE_XPR; do [ -e "$d" ] || { echo "MISSING $d"; exit 1; }; done
[ -e "$PROJ_DIR" ] && { echo "ABORT: $PROJ_DIR exists"; exit 1; }
mkdir -p $(dirname $PROJ_DIR)
vivado -mode batch -nojournal -log /tmp/vivado_256bil_make.log -source /tmp/make_project_512.tcl \
  -tclargs $BASE_XPR $PROJ_NAME $PROJ_DIR $BUILD $FINN "$IODMA_P0 $IODMA_P7" > /dev/null 2>&1
grep -E "^(===|VALIDATE_RESULT|GENERATE_RESULT|MAKE_PROJECT|UPGRADE_IP_ERROR)|^(ERROR|CRITICAL WARNING)" /tmp/vivado_256bil_make.log | cut -c1-400
P=$PROJ_DIR
echo "== refs: analytical_v1 build / bilinear build / other builds in xpr"
grep -o "finn_build_tmp/S12_dense_[A-Za-z0-9_]*" $P/$PROJ_NAME.xpr | sort | uniq -c
echo SCRIPT_DONE
