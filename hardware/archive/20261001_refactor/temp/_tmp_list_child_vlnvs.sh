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

for d in "${!DIRMAP[@]}"; do
  bdname="${DIRMAP[$d]}"
  bd="$BASE/$d/finn_vivado_stitch_proj.srcs/sources_1/bd/$bdname/$bdname.bd"
  echo "===== $d ($bdname) ====="
  if [ -f "$bd" ]; then
    grep -oP '(?<="vlnv": ")[^"]+' "$bd" | grep -vE 'interface|aximm|axis|aclk|areset' | sort -u
  else
    echo "MISSING $bd"
  fi
  echo ""
done
