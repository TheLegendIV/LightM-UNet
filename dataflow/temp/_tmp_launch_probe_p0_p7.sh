#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/probe_naming_fix_p0_p7_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"
nohup python3 finn_s12_build.py \
    finn_deployment_outputs/S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558 \
    --tag S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_probe \
    --conv-order quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json \
    --folding-json layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json \
    --partitions 0 7 --probe \
    --output-dir "$OUTDIR" \
    > "$OUTDIR/probe_build.log" 2>&1 &
echo "LAUNCHED_PID=$!"
echo "OUTDIR=$OUTDIR"
