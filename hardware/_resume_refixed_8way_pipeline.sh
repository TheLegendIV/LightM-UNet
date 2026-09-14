#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet

# Resume after container loss (Docker Desktop/WSL auto-update killed lucid_ptolemy
# mid-build on 2026-09-14). Preamble + bridge/folding-config stage already
# completed successfully and survived (bind-mounted notebooks/enet) -- skip
# straight to the real per-partition HLS/Vivado build using the EXISTING dirs.
PREAMBLE_DIR="/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_preamble_20260913_204947"
OUTDIR="/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha025_trained_8way_full_20260913_205942_refixed"
echo "PREAMBLE_DIR=$PREAMBLE_DIR"
echo "OUTDIR=$OUTDIR"

python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_trained_8way_full.py "$PREAMBLE_DIR" "$OUTDIR"
python3 finn_ooc_12_dense_relu_warmstart150ep_alpha025_8way_per_partition_synth.py "$OUTDIR"
