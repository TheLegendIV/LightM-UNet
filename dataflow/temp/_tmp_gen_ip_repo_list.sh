#!/bin/bash
set -e
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
OUT=/tmp/ip_repo_paths_list.txt
: > "$OUT"
echo "/home/thelegendiv/finn/finn-rtllib/memstream" >> "$OUT"
for p in 0 1 2 3 4 5 6 7; do
  find "$BASE/GenericPartition_${p}" -maxdepth 1 -iname 'vivado_stitch_proj_*' -type d | while read -r d; do
    echo "$d/ip" >> "$OUT"
  done
  find "$BASE/GenericPartition_${p}" -path '*code_gen_ipgen_*hls*/project_*/sol1/impl/ip' -type d >> "$OUT"
done
echo "--- wrote $(wc -l < "$OUT") lines to $OUT ---"
cat "$OUT"
