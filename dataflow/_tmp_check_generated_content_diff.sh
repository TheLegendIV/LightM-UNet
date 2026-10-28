#!/bin/bash
PROJ=/home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.gen/sources_1/bd/top/ip
for p in 0 1 2; do
  d=$(find $PROJ/top_GenericPartition_${p}_0_0 -ipath "*DuplicateStreams_hls_0*" -iname "*.v" 2>/dev/null | head -1)
  echo "--- partition $p generated DuplicateStreams_hls_0 file: $d ---"
  if [ -n "$d" ]; then
    wc -l "$d"
    md5sum "$d"
    grep -m2 -E "input|output" "$d" | head -2
  fi
done
