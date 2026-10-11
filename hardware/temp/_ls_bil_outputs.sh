#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
for d in S12_dense_256_u4_bilinear_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_2026100*; do
  echo "== $d"
  ls "$d/intermediate_models"
  ls "$d/intermediate_models/supported_op_partitions" 2>/dev/null
  ls "$d/report" | head -30
done
head -12 /tmp/ooc_S12_bilinear_full.log | cut -c1-200
grep -n "OUTPUT_DIR\|build done\|partition . done" /tmp/ooc_S12_bilinear_full.log | head -20
