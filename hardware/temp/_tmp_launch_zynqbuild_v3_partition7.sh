#!/bin/bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh
export HOME=/tmp/home_dir
export FINN_BUILD_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7
cd /home/thelegendiv/finn/notebooks/enet

PREAMBLE_DIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_preamble_20260917_234018
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=finn_deployment_outputs/zynqbuild_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_v3_partition7_${TIMESTAMP}

nohup python3 -u finn_zynqbuild_12_dense_relu_alpha025_rtl_mvau_v3_partitionN.py 7 "$PREAMBLE_DIR" "/home/thelegendiv/finn/notebooks/enet/$OUTDIR" \
    > finn_deployment_outputs/zynqbuild_rtl_mvau_v3_partition7.log 2>&1 &
echo "launched, pid $!"
echo "OUTDIR=$OUTDIR"
