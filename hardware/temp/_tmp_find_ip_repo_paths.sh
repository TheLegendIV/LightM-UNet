#!/bin/bash
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
DIRS="vivado_stitch_proj_6_75nd4k vivado_stitch_proj_4ti661rj vivado_stitch_proj_e_z5561z vivado_stitch_proj_9qynp0ng vivado_stitch_proj_ohvum343 vivado_stitch_proj_anxxap8z vivado_stitch_proj_5kxs5a6b vivado_stitch_proj_btoactkv"

for d in $DIRS; do
  echo "== $d =="
  xpr=$(find "$BASE/$d" -maxdepth 1 -iname "*.xpr" 2>/dev/null | head -1)
  echo "xpr: $xpr"
  if [ -n "$xpr" ]; then
    grep -o 'ip_repo_paths[^/]*"[^"]*"' "$xpr" | head -5
    grep -oP '(?<=<Option Name="IPRepoPaths" Val=")[^"]+' "$xpr" | head -5
  fi
done
