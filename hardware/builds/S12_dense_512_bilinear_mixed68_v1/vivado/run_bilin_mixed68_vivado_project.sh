#!/bin/bash
# Runs inside finn_persistent. Copy of s12_512_analytical whose FINN partition IPs come from the
# bilinear mixed68 build. This build has no IODMA IPs, so the repackaged p0_input/p7_output IODMA
# IPs from the 512 analytical build are kept (same I/O geometry: UINT8 in, UINT8 class map out).
# Needs /tmp/make_project_512.tcl (hardware/builds/S12_dense_512_u4_analytical_v1/vivado/).
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
BUILD=$FINN/notebooks/enet/finn_build_tmp/S12_dense_512_bilinear_mixed68_v1_ft15ep_dw
OLD=$FINN/notebooks/enet/finn_build_tmp/S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200
IODMA_P0=$OLD/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_td5gxqso/project_IODMA_hls_0/sol1/impl/ip
IODMA_P7=$OLD/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_uv5wu1iw/project_IODMA_hls_0/sol1/impl/ip
BASE_XPR=$FINN/vivado_projects/S12_512_analytical/s12_512_analytical/s12_512_analytical.xpr
PROJ_NAME=s12_512_bilin_mixed68
PROJ_DIR=$FINN/vivado_projects/S12_512_bilin_mixed68/$PROJ_NAME
cd /tmp
for d in $BUILD $IODMA_P0/component.xml $IODMA_P7/component.xml $BASE_XPR /tmp/make_project_512.tcl; do [ -e "$d" ] || { echo "MISSING $d"; exit 1; }; done
[ -e "$PROJ_DIR" ] && { echo "ABORT: $PROJ_DIR exists"; exit 1; }
mkdir -p $(dirname $PROJ_DIR)
vivado -mode batch -nojournal -log /tmp/vivado_bilin_mixed68_make.log -source /tmp/make_project_512.tcl \
  -tclargs $BASE_XPR $PROJ_NAME $PROJ_DIR $BUILD $FINN "$IODMA_P0 $IODMA_P7" > /dev/null 2>&1
grep -E "^(===|VALIDATE_RESULT|GENERATE_RESULT|MAKE_PROJECT|UPGRADE_IP_ERROR)|^(ERROR|CRITICAL WARNING)" /tmp/vivado_bilin_mixed68_make.log | cut -c1-400
P=$PROJ_DIR
echo "== refs to finn_build_tmp builds in xpr"
grep -o "finn_build_tmp/S12_dense_[A-Za-z0-9_]*" $P/$PROJ_NAME.xpr | sort | uniq -c
echo SCRIPT_DONE
