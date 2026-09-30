#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
DIRS="vivado_stitch_proj_6_75nd4k vivado_stitch_proj_4ti661rj vivado_stitch_proj_e_z5561z vivado_stitch_proj_9qynp0ng vivado_stitch_proj_ohvum343 vivado_stitch_proj_anxxap8z vivado_stitch_proj_5kxs5a6b vivado_stitch_proj_btoactkv"

for d in $DIRS; do
  p="$BASE/$d"
  echo "===== $d ====="
  bd=$(find "$p" -iname "*.bd" 2>/dev/null | head -1)
  echo "bd file: $bd"
  if [ -n "$bd" ]; then
    echo "cell VLNVs referenced inside bd:"
    grep -oP '(?<=CONFIG.EDIF_VLNV \{)[^}]+|(?<=CELL_VLNV \{)[^}]+' "$bd" 2>/dev/null | sort -u | head -30
    echo "-- raw grep for xilinx_finn:finn: refs --"
    grep -oP 'xilinx_finn:finn:[A-Za-z0-9_.]+:[0-9.]+' "$bd" 2>/dev/null | sort -u | head -30
  fi
  echo ""
done
