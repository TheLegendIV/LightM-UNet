#!/bin/bash
P0=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip/component.xml
P7=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip/component.xml
echo "--- P0 component name ---"
grep -o '<spirit:name>[^<]*' "$P0" | head -3
echo "--- P7 component name ---"
grep -o '<spirit:name>[^<]*' "$P7" | head -3
