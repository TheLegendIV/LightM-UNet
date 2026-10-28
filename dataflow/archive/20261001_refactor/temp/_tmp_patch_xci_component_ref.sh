#!/bin/bash
# The .bd JSON's per-cell "vlnv" field was already patched to the renamed
# VLNV, but Vivado actually resolves each cell's IP identity from the
# separate cached .xci file's "component_reference" field (there are 2
# copies per cell: proj/ip/src/<cell>/*.xci and
# proj/<proj>.srcs/sources_1/bd/<bd>/ip/<cell>/*.xci). Those still point at
# the OLD, unrenamed VLNV, causing Vivado to report "IP definition not
# found" / locked cells. Patch component_reference in both copies to match.
set -e
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
RENAME_TCL="$BASE/rename_catalog_ips_for_combine.tcl"

declare -A DIRMAP=(
  [0]=vivado_stitch_proj_6_75nd4k
  [1]=vivado_stitch_proj_4ti661rj
  [2]=vivado_stitch_proj_e_z5561z
  [3]=vivado_stitch_proj_9qynp0ng
  [4]=vivado_stitch_proj_ohvum343
  [5]=vivado_stitch_proj_anxxap8z
  [6]=vivado_stitch_proj_5kxs5a6b
  [7]=vivado_stitch_proj_btoactkv
)

total=0
# lines look like: "# partition 0: DuplicateStreams_hls_0 -> p0_DuplicateStreams_hls_0"
grep -E '^# partition [0-9]+: ' "$RENAME_TCL" | while read -r line; do
  rest="${line#\# partition }"
  idx="${rest%%:*}"
  rest2="${rest#*: }"
  oldname="${rest2%% -> *}"
  newname="${rest2#* -> }"
  proj_dir="$BASE/${DIRMAP[$idx]}"
  old_ref="\"component_reference\": \"xilinx.com:hls:${oldname}:1.0\""
  new_ref="\"component_reference\": \"xilinx.com:hls:${newname}:1.0\""
  count=0
  while IFS= read -r -d '' xci; do
    if grep -qF "$old_ref" "$xci"; then
      python3 - "$xci" "$old_ref" "$new_ref" <<'PYEOF'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
s2 = s.replace(old, new)
if s2 != s:
    open(p, "w").write(s2)
PYEOF
      count=$((count+1))
    fi
  done < <(find "$proj_dir" -iname "*.xci" -print0)
  echo "partition $idx: $oldname -> $newname : patched $count xci file(s)"
done
echo "DONE"
