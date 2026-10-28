#!/bin/bash
set -e
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
ARCH=$BASE/archive_pre_20260920_combine_rename
count=0
for d in $ARCH/p*_code_gen_ipgen_*; do
  bn=$(basename "$d")
  orig=$(echo "$bn" | sed -E 's/^p[0-9]+_//')
  rm -rf "$BASE/$orig"
  cp -r "$d" "$BASE/$orig"
  count=$((count+1))
done
echo "RESTORED_COUNT=$count"

IARCH=$BASE/archive_pre_20260920_iodma_rename
rm -rf "$BASE/code_gen_ipgen_StreamingDataflowPartition_0_IODMA_hls_0_66qk80st"
cp -r "$IARCH/idma_zynqp0_66qk80st_orig" "$BASE/code_gen_ipgen_StreamingDataflowPartition_0_IODMA_hls_0_66qk80st"
rm -rf "$BASE/zynqbuild_partition7/code_gen_ipgen_StreamingDataflowPartition_2_IODMA_hls_0_pu7nilya"
cp -r "$IARCH/odma_zynqp7_pu7nilya_orig" "$BASE/zynqbuild_partition7/code_gen_ipgen_StreamingDataflowPartition_2_IODMA_hls_0_pu7nilya"
echo "RESTORED_IODMA"
