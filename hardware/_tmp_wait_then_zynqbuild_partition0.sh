#!/bin/bash
# Waits for the currently-running main 8-way per-partition OOC synth build
# (PID 2739816, finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_
# partition_synth.py) to finish, then launches the real ZynqBuild (bitstream
# + PYNQ driver) for partition 0 of the SAME rtl_mvau architecture.
set -e
MAIN_BUILD_PID=2739816
echo "Waiting for main build PID $MAIN_BUILD_PID to finish..."
while kill -0 "$MAIN_BUILD_PID" 2>/dev/null; do
    sleep 30
done
echo "Main build PID $MAIN_BUILD_PID finished, launching ZynqBuild for partition 0."

cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
PREAMBLE_DIR=/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_preamble_20260917_005512
OUTDIR=finn_deployment_outputs/zynqbuild_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_partition0_$(date +%Y%m%d_%H%M%S)
nohup python3 -u finn_zynqbuild_12_dense_relu_alpha025_rtl_mvau_partition0.py "$PREAMBLE_DIR" "/home/thelegendiv/finn/notebooks/enet/$OUTDIR" \
    > /tmp/zynqbuild_12_dense_relu_alpha025_rtl_mvau_partition0.log 2>&1 &
echo "LAUNCHED_PID=$!"
echo "OUTDIR=$OUTDIR"
