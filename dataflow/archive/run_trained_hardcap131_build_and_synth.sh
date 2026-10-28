#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
OUTDIR="finn_deployment_outputs/hawq_12_sep_dense_relu_min4_trained_hardcap131_8way_full_$(date +%Y%m%d_%H%M%S)"
echo "OUTDIR=$OUTDIR"
python3 finn_ooc_12_separable_dense_relu_min4_trained_hardcap131_8way_full.py \
    finn_deployment_outputs/hawq_12_sep_dense_relu_min4_trained_preamble_20260903_230918 \
    /home/thelegendiv/finn/notebooks/enet/$OUTDIR
python3 finn_ooc_12_separable_dense_relu_min4_8way_per_partition_synth.py \
    /home/thelegendiv/finn/notebooks/enet/$OUTDIR
