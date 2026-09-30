#!/bin/bash
cd "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/12_dense_relu_warmstart150ep_alpha05_trained_rtl_mvau_8way_full_v4_512x512_20260923_223139/report"
for f in ooc_synth_partition_0.json ooc_synth_partition_1.json ooc_synth_partition_2.json ooc_synth_partition_3.json ooc_synth_partition_4.json ooc_synth_partition_5_refix.json ooc_synth_partition_7.json; do
  echo "=== $f ==="
  cat "$f"
  echo
done
