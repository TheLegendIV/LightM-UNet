#!/bin/bash
set -e
cd /home/thelegendiv/finn/notebooks/enet
for TAG in dsrSweep_pbiOff_dsr1.5 dsrSweep_pbiOff_dsr3.0 dsrSweep_pbiOff_dsr5.0 pbiSweep_dsrOff_pbi1.5; do
  echo "===== $TAG preamble ====="
  python3 finn_hawq_preamble_trained.py "$TAG" > "/tmp/preamble_${TAG}.log" 2>&1
  echo "exit=$?"
  PDIR=$(grep -oP '(?<=^OUTPUT_DIR= ).*' "/tmp/preamble_${TAG}.log" | tail -1)
  echo "preamble dir: $PDIR"
  echo "$TAG $PDIR" >> /tmp/preamble_dirs.txt
  tail -5 "/tmp/preamble_${TAG}.log"
  echo "===== $TAG bridge ====="
  python3 finn_hawq_folding_bridge_nearest_upsample.py "$PDIR" "layer_bits_folding_${TAG}.json" > "/tmp/bridge_${TAG}.log" 2>&1
  echo "exit=$?"
  tail -25 "/tmp/bridge_${TAG}.log"
done
