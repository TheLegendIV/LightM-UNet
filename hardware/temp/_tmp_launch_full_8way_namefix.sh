#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR=finn_deployment_outputs/S12_256_analytical_namefix_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"
nohup python3 finn_s12_build.py \
    finn_deployment_outputs/S12_dense_256_u4_analytical_v1_ft15ep_preamble_20261005_220558 \
    --tag S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix \
    --conv-order quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_conv_order.json \
    --folding-json layer_bits_folding_S12_dense_256_u4_analytical_v1_int6_fps250_lat200_20261006.json \
    --partitions all \
    --output-dir "$OUTDIR" \
    > "$OUTDIR/full_build.log" 2>&1 &
echo "LAUNCHED_PID=$!"
echo "OUTDIR=$OUTDIR"
