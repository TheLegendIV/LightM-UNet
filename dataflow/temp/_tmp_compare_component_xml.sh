#!/bin/bash
IODMA_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip
MAXPOOL_DIR=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix/GenericPartition_1/code_gen_ipgen_GenericPartition_1_StreamingMaxPool_hls_0_6qx04yf8/project_GenericPartition_1_StreamingMaxPool_hls_0/sol1/impl/ip
echo "=== IODMA component.xml (old build) key tags ==="
find "$IODMA_DIR" -iname 'component.xml' -exec grep -i 'vendor>\|library>\|<spirit:name>\|version>\|taxonomy\|coreRevision\|hlsTargetLanguage\|vitis' {} \;
echo ""
echo "=== StreamingMaxPool component.xml (current build) key tags ==="
find "$MAXPOOL_DIR" -iname 'component.xml' -exec grep -i 'vendor>\|library>\|<spirit:name>\|version>\|taxonomy\|coreRevision\|hlsTargetLanguage\|vitis' {} \;
