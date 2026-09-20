#!/bin/bash
# Patch each partition's OWN top-level .bd JSON so its cell references point
# at the newly-renamed (partition-prefixed) catalog IP VLNVs, instead of the
# old, collision-prone bare names. Backs up each .bd file first.
#
# Must be run AFTER _tmp_gen_rename_catalog_ips.sh has produced
# rename_catalog_ips_for_combine.tcl (we reuse its "# partition N: OLD -> NEW"
# comment lines as the substitution manifest, so the two scripts never
# disagree about what got renamed).
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
BACKUP_DIR="$BASE/archive_pre_20260920_combine_rename"
RENAME_TCL="$BASE/rename_catalog_ips_for_combine.tcl"
mkdir -p "$BACKUP_DIR"

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

for idx in 0 1 2 3 4 5 6 7; do
  d="${DIRMAP[$idx]}"
  proj_dir="$BASE/$d"
  bdname="StreamingDataflowPartition_${idx}"
  bd="$proj_dir/finn_vivado_stitch_proj.srcs/sources_1/bd/$bdname/$bdname.bd"
  if [ ! -f "$bd" ]; then
    echo "== partition $idx: BD NOT FOUND at $bd, skipping =="
    continue
  fi
  echo "== partition $idx ($d): patching $bd =="
  cp "$bd" "$BACKUP_DIR/p${idx}_${bdname}.bd.orig"

  # pull this partition's OLD -> NEW typename pairs from the rename manifest
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
    fi
  done
done

echo ""
echo "Backups of original .bd files: $BACKUP_DIR/p<idx>_StreamingDataflowPartition_<idx>.bd.orig"
