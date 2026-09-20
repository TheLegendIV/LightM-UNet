#!/bin/bash
# Adapted from hardware/temp/_tmp_patch_xci_component_ref.sh for the tiny 2-partition test.
set -e
BASE=/tmp/finn_dev_thelegendiv
RENAME_TCL="$BASE/tiny_rename_catalog_ips_for_combine.tcl"

declare -A DIRMAP=(
  [0]=vivado_stitch_proj_hf0tokm3
  [1]=vivado_stitch_proj_rtj421tm
)

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
