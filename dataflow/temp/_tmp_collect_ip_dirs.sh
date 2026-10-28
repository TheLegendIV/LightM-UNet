#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
for p in 0 1 2 3 4 5 6 7; do
  echo "PARTITION_${p}:"
  find "$BASE/GenericPartition_${p}" -maxdepth 1 -iname 'vivado_stitch_proj_*' -type d
  find "$BASE/GenericPartition_${p}" -path '*code_gen_ipgen_*hls*/project_*/sol1/impl/ip' -type d
done
