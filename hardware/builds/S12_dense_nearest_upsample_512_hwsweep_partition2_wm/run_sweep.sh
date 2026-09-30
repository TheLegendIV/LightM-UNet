#!/bin/bash
# Run inside the FINN container (after docker cp-ing in, flat into
# /home/thelegendiv/finn/notebooks/enet/):
#   - the 5 trained onnx exports (quantEnet_12_dense_relu_nearest_upsample_trained_<tag>_512x512.onnx)
#   - quantEnet_12_dense_relu_nearest_upsample_dummy_int8_conv_order.json
#   - finn_hawq_preamble_trained.py, finn_hawq_folding_bridge_nearest_upsample.py,
#     finn_ooc_partition2_trained.py
#   - each MILP/artifacts/.../<tag>/layer_bits_folding_<tag>.json
#
# Runs the cheap (no-Vivado) preamble + folding-bridge steps sequentially per
# tag, then launches all 6 partition-2 OOC syntheses in parallel background
# workers (mirrors hardware/temp/run_partitions_3456_parallel.sh's proven
# nohup pattern -- that ran 4 concurrent partition builds successfully).
set -e
cd /home/thelegendiv/finn/notebooks/enet
source /tools/Xilinx/Vivado/2022.2/settings64.sh > /dev/null 2>&1

TAGS="baseline_both_off dsrSweep_pbiOff_dsr1.5 dsrSweep_pbiOff_dsr3.0 dsrSweep_pbiOff_dsr5.0 pbiSweep_dsrOff_pbi1.5"
rm -f /tmp/preamble_dirs.txt

for TAG in $TAGS; do
  echo "=== [$TAG] preamble ==="
  python3 finn_hawq_preamble_trained.py "$TAG" 2>&1 | tee "/tmp/preamble_${TAG}.log"
  PDIR=$(grep -oP '(?<=^OUTPUT_DIR= ).*' "/tmp/preamble_${TAG}.log" | tail -1)
  echo "preamble dir for $TAG: $PDIR"
  echo "$TAG $PDIR" >> /tmp/preamble_dirs.txt

  echo "=== [$TAG] folding bridge ==="
  FOLDING_JSON="/home/thelegendiv/finn/notebooks/enet/layer_bits_folding_${TAG}.json"
  python3 finn_hawq_folding_bridge_nearest_upsample.py "$PDIR" "$FOLDING_JSON" 2>&1 | tee "/tmp/bridge_${TAG}.log"
done

echo "=== launching 6 OOC syntheses in parallel (MILP-fold x5 + auto-fold x1) ==="
while read -r TAG PDIR; do
  LOG="/tmp/ooc_${TAG}_milpfold.log"
  nohup python3 finn_ooc_partition2_trained.py "$PDIR" "$TAG" "${PDIR}/hawq_folding_config_partition2.json" > "$LOG" 2>&1 &
  echo "LAUNCHED $TAG (milp-fold) PID=$!"
done < /tmp/preamble_dirs.txt

BASELINE_PDIR=$(grep '^baseline_both_off ' /tmp/preamble_dirs.txt | awk '{print $2}')
LOG="/tmp/ooc_baseline_both_off_autofold.log"
nohup python3 finn_ooc_partition2_trained.py "$BASELINE_PDIR" "baseline_both_off" > "$LOG" 2>&1 &
echo "LAUNCHED baseline_both_off (auto-fold) PID=$!"

echo "All 6 launched. Tail logs with: tail -f /tmp/ooc_*.log"
