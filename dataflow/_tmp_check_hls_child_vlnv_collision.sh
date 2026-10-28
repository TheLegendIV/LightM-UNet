#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200
for i in 0 1 2; do
  f=$(ls $BASE/GenericPartition_$i/code_gen_ipgen_DuplicateStreams_hls_0_*/project_DuplicateStreams_hls_0/sol1/impl/ip/component.xml 2>/dev/null | head -1)
  echo "--- partition $i DuplicateStreams_hls_0 : $f ---"
  grep -E "spirit:(vendor|library|name|version)>" "$f" | head -4
  echo "--- same partition $i, hash/content check (md5 of synth verilog) ---"
  md5sum $BASE/GenericPartition_$i/code_gen_ipgen_DuplicateStreams_hls_0_*/project_DuplicateStreams_hls_0/sol1/impl/ip/hdl/ip/*/hdl/verilog/*.v 2>/dev/null | head -3
done
