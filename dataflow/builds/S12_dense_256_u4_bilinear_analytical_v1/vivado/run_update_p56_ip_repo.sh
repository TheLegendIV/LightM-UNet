#!/bin/bash
# Runs inside finn_persistent.
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
FINN=/home/thelegendiv/finn
BUILD=$FINN/notebooks/enet/finn_build_tmp/S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200
XPR=$FINN/vivado_projects/S12_256_bilinear_analytical/s12_256_bilinear_analytical/s12_256_bilinear_analytical.xpr
[ -e "$XPR" ] || { echo "MISSING $XPR"; exit 1; }
cp -n $XPR $XPR.bak_pre_p56_dwfix
cd /tmp
vivado -mode batch -nojournal -log /tmp/vivado_256bil_p56_repo.log -source /tmp/update_p56_ip_repo.tcl -tclargs $XPR $BUILD > /dev/null 2>&1
grep -E "^(===|DROP|VALIDATE_RESULT|GENERATE_RESULT|UPGRADE_IP_ERROR|REPO_UPDATE_DONE)|^(ERROR|CRITICAL WARNING)" /tmp/vivado_256bil_p56_repo.log | cut -c1-300
echo "== xpr refs P5/P6 stitch"
grep -o "GenericPartition_[56]/vivado_stitch_proj_[a-z0-9_]*" $XPR | sort | uniq -c
echo SCRIPT_DONE
