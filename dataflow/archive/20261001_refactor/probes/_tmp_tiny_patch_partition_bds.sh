#!/bin/bash
# Adapted from hardware/temp/_tmp_patch_partition_bds.sh for the tiny 2-partition test.
BASE=/tmp/finn_dev_thelegendiv
BACKUP_DIR="$BASE/archive_pre_20260920_tiny_combine_test"
RENAME_TCL="$BASE/tiny_rename_catalog_ips_for_combine.tcl"
mkdir -p "$BACKUP_DIR"

declare -A DIRMAP=(
  [0]=vivado_stitch_proj_hf0tokm3
  [1]=vivado_stitch_proj_rtj421tm
)

for idx in 0 1; do
  d="${DIRMAP[$idx]}"
  proj_dir="$BASE/$d"
  bdname="finn_design"
  bd="$proj_dir/finn_vivado_stitch_proj.srcs/sources_1/bd/$bdname/$bdname.bd"
  if [ ! -f "$bd" ]; then
    echo "== partition $idx: BD NOT FOUND at $bd, skipping =="
    continue
  fi
  echo "== partition $idx ($d): patching $bd =="
  cp "$bd" "$BACKUP_DIR/p${idx}_${bdname}.bd.orig"

  grep -E "^# partition $idx: " "$RENAME_TCL" | sed -E "s/^# partition $idx: //" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    old="${line%% -> *}"
    new="${line#* -> }"
    [ -z "$old" ] && continue
    [ -z "$new" ] && continue
    old_vlnv="xilinx.com:hls:${old}:1.0"
    new_vlnv="xilinx.com:hls:${new}:1.0"
    if grep -qF "\"$old_vlnv\"" "$bd"; then
      sed -i "s|\"$old_vlnv\"|\"$new_vlnv\"|g" "$bd"
      echo "  patched: $old_vlnv -> $new_vlnv"
    else
      echo "  NOT FOUND in bd: $old_vlnv"
    fi
  done
done
echo "DONE"
