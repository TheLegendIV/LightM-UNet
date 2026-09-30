#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
PREAMBLE=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_512x512_preamble_20260920_180600
OUTDIR=finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram_512x512_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"
nohup python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_rtl_mvau_8way_full_v3_nouram.py "$PREAMBLE" "/home/thelegendiv/finn/notebooks/enet/$OUTDIR" > "$OUTDIR/build_and_synth.log" 2>&1 &
disown
echo "LAUNCHED OUTDIR=$OUTDIR"
