#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
declare -A DIRMAP=(
  [vivado_stitch_proj_6_75nd4k]=StreamingDataflowPartition_0
  [vivado_stitch_proj_4ti661rj]=StreamingDataflowPartition_1
  [vivado_stitch_proj_e_z5561z]=StreamingDataflowPartition_2
  [vivado_stitch_proj_9qynp0ng]=StreamingDataflowPartition_3
  [vivado_stitch_proj_ohvum343]=StreamingDataflowPartition_4
  [vivado_stitch_proj_anxxap8z]=StreamingDataflowPartition_5
  [vivado_stitch_proj_5kxs5a6b]=StreamingDataflowPartition_6
  [vivado_stitch_proj_btoactkv]=StreamingDataflowPartition_7
)

for d in vivado_stitch_proj_6_75nd4k vivado_stitch_proj_4ti661rj vivado_stitch_proj_e_z5561z vivado_stitch_proj_9qynp0ng vivado_stitch_proj_ohvum343 vivado_stitch_proj_anxxap8z vivado_stitch_proj_5kxs5a6b vivado_stitch_proj_btoactkv; do
  bdname="${DIRMAP[$d]}"
  bd="$BASE/$d/finn_vivado_stitch_proj.srcs/sources_1/bd/$bdname/$bdname.bd"
  echo "== $d ($bdname) =="
  if [ -f "$bd" ]; then
    grep -oP '(?<="vlnv": ")[^"]+' "$bd" | grep -E 'hls:|amd\.com:finn|xilinx_finn' | sed -E 's/_[0-9]+:1\.0$/_N:1.0/' | sort -u
  else
    echo "MISSING"
  fi
done
