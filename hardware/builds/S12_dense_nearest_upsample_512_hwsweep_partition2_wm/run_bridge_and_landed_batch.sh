#!/bin/bash
set -e
for TAG in dsr_off dsrmin_1x dsrmin_4x dsrmin_8x; do
  PRE=$(ls -d /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/*${TAG}_preamble* | head -1)
  cd /home/thelegendiv/finn/notebooks/enet
  echo "=== $TAG (preamble=$PRE) ==="
  python3 finn_hawq_folding_bridge_nearest_upsample.py $PRE layer_bits_folding_${TAG}.json > /tmp/bridge_${TAG}.log 2>&1
  tail -5 /tmp/bridge_${TAG}.log
  python3 dump_milpfold_landed_partition2.py $PRE $PRE/hawq_folding_config_partition2.json /home/thelegendiv/finn/notebooks/enet/landed_partition2_${TAG}_milpfold.onnx > /tmp/dump_${TAG}.log 2>&1
  tail -3 /tmp/dump_${TAG}.log
done
