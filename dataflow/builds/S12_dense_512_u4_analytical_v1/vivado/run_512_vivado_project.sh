#!/bin/bash
# Runs inside finn_persistent. Prepares 512 IODMA IPs then creates the 512 Vivado project.
# Usage: docker exec -e HOME=/tmp/home_dir finn_persistent bash /tmp/run_512_vivado_project.sh
source /tools/Xilinx/Vivado/2022.2/settings64.sh
set -u
BUILD=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_512_u4_analytical_v1_ft15ep_int6_fps250_lat200
RENAMED=/home/thelegendiv/finn/vivado_projects/S12_512_analytical_srcs/iodma_renamed
BASE_XPR=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
PROJ_NAME=s12_512_analytical
PROJ_DIR=/home/thelegendiv/finn/vivado_projects/S12_512_analytical/$PROJ_NAME
LOG=/tmp/vivado_512_project.log
cd /tmp
: > $LOG
if [ -e "$PROJ_DIR" ]; then echo "ABORT: $PROJ_DIR exists" | tee -a $LOG; exit 1; fi
mkdir -p $RENAMED $(dirname $PROJ_DIR)
python3 /tmp/rename_iodma_sources_512.py $BUILD $RENAMED 2>&1 | tee -a $LOG
vivado -mode batch -nojournal -log /tmp/vivado_512_prep.log -source /tmp/prep_iodma_512.tcl -tclargs $BUILD $RENAMED > /dev/null 2>&1
grep -E "^\[p|PREP_IODMA|^ERROR" /tmp/vivado_512_prep.log | tee -a $LOG
vivado -mode batch -nojournal -log /tmp/vivado_512_make.log -source /tmp/make_project_512.tcl -tclargs $BASE_XPR $PROJ_NAME $PROJ_DIR $BUILD > /dev/null 2>&1
grep -E "^(===|VALIDATE_RESULT|GENERATE_RESULT|MAKE_PROJECT|UPGRADE_IP_ERROR)|^(ERROR|CRITICAL WARNING)" /tmp/vivado_512_make.log | cut -c1-400 | tee -a $LOG
echo SCRIPT_DONE | tee -a $LOG
