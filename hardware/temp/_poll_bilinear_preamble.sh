#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs
for d in $(ls -d S12_dense_256_u4_analytical_v*_preamble_* 2>/dev/null | tail -2); do
  echo "== $d"
  cut -c1-120 $d/build_dataflow.log | head -8
done
echo "== bilinear"
d=$(ls -d S12_dense_256_u4_bilinear_analytical_v1_ft15ep_preamble_* | tail -1)
cut -c1-150 $d/build_dataflow.log | tail -4
date +%H:%M:%S
