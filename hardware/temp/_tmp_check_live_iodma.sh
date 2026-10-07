#!/bin/bash
BUILD=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
OLD_BUILD=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200
echo "=== IODMA dirs under CURRENT (_namefix) build ==="
find "$BUILD" -maxdepth 2 -iname '*IODMA*'
echo "=== does OLD (non-namefix, registered in xpr) build dir even still exist? ==="
if [ -d "$OLD_BUILD" ]; then echo "EXISTS: $OLD_BUILD"; find "$OLD_BUILD" -maxdepth 2 -iname '*IODMA*'; else echo "MISSING: $OLD_BUILD (stale ip_repo_path, dangling)"; fi
echo "=== component.xml taxonomy/vendor tags for a known-good HLS IP in current build (StreamingMaxPool_hls) ==="
find "$BUILD/GenericPartition_1" -maxdepth 1 -iname 'code_gen_ipgen*StreamingMaxPool_hls_0*' | head -1
