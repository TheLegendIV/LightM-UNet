#!/bin/bash
BUILD=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_namefix
echo "=== GenericPartition_0 codegen dirs (non-hls only) ==="
find "$BUILD/GenericPartition_0" -maxdepth 1 -type d -iname 'code_gen_ipgen_*' ! -iname '*_hls_*'
echo "=== does any contain a component.xml directly (IP-XACT packaged)? ==="
for d in $(find "$BUILD/GenericPartition_0" -maxdepth 1 -type d -iname 'code_gen_ipgen_*' ! -iname '*_hls_*'); do
  if [ -f "$d/component.xml" ]; then
    echo "HAS component.xml: $d"
  else
    echo "no component.xml: $d"
  fi
done
