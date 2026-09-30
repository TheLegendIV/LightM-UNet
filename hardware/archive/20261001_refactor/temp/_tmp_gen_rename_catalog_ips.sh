#!/bin/bash
# Discover per-partition catalog HLS child IPs that collide across partitions
# (memstream is intentionally excluded -- it's ONE shared generic IP from
# finn-rtllib, reused everywhere by design, not a per-node export), back
# each one up, and emit one big Tcl script that renames them all with a
# partition-specific prefix.
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
BACKUP_DIR="$BASE/archive_pre_20260920_combine_rename"
TCL_OUT="$BASE/rename_catalog_ips_for_combine.tcl"
mkdir -p "$BACKUP_DIR"
cat > "$TCL_OUT" <<'EOF'
# ipx::edit_ip_in_project requires an already-open project as context.
create_project -force scratch_rename_proj /tmp/scratch_rename_proj -part xczu7ev-ffvc1156-2-e

EOF

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

HLS_PATTERN='code_gen_ipgen_(DuplicateStreams_hls|AddStreams_hls|VVAU_hls|FMPadding_Pixel_hls|UpsampleNearestNeighbour_hls|StreamingMaxPool_hls|StreamingConcat_hls|ChannelwiseOp_hls)_[0-9]+_'

for idx in 0 1 2 3 4 5 6 7; do
  d="${DIRMAP[$idx]}"
  proj_dir="$BASE/$d"
  xpr=$(find "$proj_dir" -maxdepth 1 -iname "*.xpr" | head -1)
  echo "== partition $idx ($d) =="
  # extract IPRepoPath values, resolve $PPRDIR to proj_dir
  grep -oP '(?<=<Option Name="IPRepoPath" Val=")[^"]+' "$xpr" | while read -r rel; do
    abs=$(echo "$rel" | sed "s|\$PPRDIR|$proj_dir|")
    abs=$(python3 -c "import os,sys; print(os.path.normpath(sys.argv[1]))" "$abs")
    base=$(basename "$abs")
    if echo "$abs" | grep -qE "$HLS_PATTERN"; then
      # node type name = strip the code_gen_ipgen_ prefix + trailing _<hash>
      typedir=$(echo "$abs" | grep -oE "code_gen_ipgen_[A-Za-z0-9_]+_hls_[0-9]+" | head -1)
      typename=$(echo "$typedir" | sed -E 's/^code_gen_ipgen_//')
      component_xml="$abs/component.xml"
      # back up the WHOLE code_gen_ipgen_* dir (2 levels up from component.xml's
      # .../project_X/sol1/impl/ip), not just the ip/ leaf
      cogen_root=$(echo "$abs" | grep -oE ".*code_gen_ipgen_[A-Za-z0-9_]+_hls_[0-9]+_[A-Za-z0-9]+" || true)
      if [ -n "$cogen_root" ] && [ -d "$cogen_root" ] && [ ! -d "$BACKUP_DIR/p${idx}_$(basename "$cogen_root")" ]; then
        cp -r "$cogen_root" "$BACKUP_DIR/p${idx}_$(basename "$cogen_root")" || echo "  WARN: backup failed for $cogen_root"
        echo "  backed up: $(basename "$cogen_root")"
      fi
      newname="p${idx}_${typename}"
      cat >> "$TCL_OUT" <<EOF
# partition $idx: $typename -> $newname
ipx::edit_ip_in_project -upgrade true -name tmp_edit_$newname -directory /tmp/ipedit_$newname {$component_xml}
set core [ipx::current_core]
set_property NAME $newname \$core
set_property VLNV [regsub {:[^:]+:[^:]+\$} [get_property VLNV \$core] ":$newname:1.0"] \$core
# Component_Name drives the generated wrapper/netlist name -- must match NAME
# or output-product generation fails with "Cannot upgrade to invalid target ''"
set_property value {${newname}_v1_0} [ipx::get_user_parameters Component_Name -of_objects \$core]
ipx::create_xgui_files \$core
ipx::update_checksums \$core
ipx::save_core \$core
close_project -delete

EOF
      echo "  queued rename: $typename -> $newname ($component_xml)"
    fi
  done
done

echo "close_project" >> "$TCL_OUT"

echo ""
echo "Backup dir: $BACKUP_DIR"
echo "Generated Tcl: $TCL_OUT"
