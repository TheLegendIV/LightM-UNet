#!/bin/bash
cd /home/thelegendiv/finn/notebooks/enet
export HOME=/tmp/home_dir
for m in quantEnet_S12_dense_256_u4_analytical_v1_finn_calibrated_ft15ep_u8in quantEnet_S12_dense_256_u4_bilinear_analytical_v1_ft15ep_u8in; do
  echo "##### $m"
  timeout 300 python3 /tmp/_diag_fold.py $m 2>&1 | grep -v Warning | tail -15
done
