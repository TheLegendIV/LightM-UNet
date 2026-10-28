#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200
for i in 0 1 2 3 4 5 6 7; do
  echo "--- partition $i ---"
  ls -d $BASE/GenericPartition_$i/code_gen_ipgen_*_hls_*/ 2>/dev/null
done
