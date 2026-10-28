#!/bin/bash
F0=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip/component.xml
F7=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip/component.xml
for f in "$F0" "$F7"; do
  echo "=== $f ==="
  ls -la "$f" 2>&1
  grep -m1 -E "spirit:vendor>|spirit:library>|spirit:name>|spirit:version>" "$f" 2>&1
  echo "--- xilinx_finn / StreamingDataflowPartition refs ---"
  grep -c "StreamingDataflowPartition\|xilinx_finn" "$f" 2>&1
  echo "--- any busInterface definitions ---"
  grep -c "busInterface" "$f" 2>&1
done
