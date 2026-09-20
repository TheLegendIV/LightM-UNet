#!/bin/bash
# Adapted from hardware/temp/_tmp_gen_rename_catalog_ips.sh, parameterized for
# the tiny 2-partition collision-test probe (p0=idx0, p1=idx1) instead of the
# hardcoded 8-way DIRMAP. Same HLS_PATTERN (HLS-backend child IPs only, as
# originally written) -- deliberately NOT extended to cover RTL-backend
# StreamingFIFO_rtl/StreamingDataWidthConverter_rtl/MVAU_rtl/etc, to validate
# the EXISTING documented fix exactly as-is first.
set -e
BASE=/tmp/finn_dev_thelegendiv
BACKUP_DIR="$BASE/archive_pre_20260920_tiny_combine_test"
TCL_OUT="$BASE/tiny_rename_catalog_ips_for_combine.tcl"
mkdir -p "$BACKUP_DIR"
cat > "$TCL_OUT" <<'EOF'
create_project -force scratch_rename_proj /tmp/scratch_rename_proj_tiny -part xczu7ev-ffvc1156-2-e

EOF

declare -A DIRMAP=(
  [0]=vivado_stitch_proj_hf0tokm3
  [1]=vivado_stitch_proj_rtj421tm
)

HLS_PATTERN='code_gen_ipgen_(DuplicateStreams_hls|AddStreams_hls|VVAU_hls|FMPadding_Pixel_hls|UpsampleNearestNeighbour_hls|StreamingMaxPool_hls|StreamingConcat_hls|ChannelwiseOp_hls|MVAU_hls)_[0-9]+_'

for idx in 0 1; do
  d="${DIRMAP[$idx]}"
  proj_dir="$BASE/$d"
  xpr=$(find "$proj_dir" -maxdepth 1 -iname "*.xpr" | head -1)
  echo "== partition $idx ($d) =="
  grep -oP '(?<=<Option Name="IPRepoPath" Val=")[^"]+' "$xpr" | while read -r rel; do
    abs=$(echo "$rel" | sed "s|\$PPRDIR|$proj_dir|")
    abs=$(python3 -c "import os,sys; print(os.path.normpath(sys.argv[1]))" "$abs")
    if echo "$abs" | grep -qE "$HLS_PATTERN"; then
      typedir=$(echo "$abs" | grep -oE "code_gen_ipgen_[A-Za-z0-9_]+_hls_[0-9]+" | head -1)
      typename=$(echo "$typedir" | sed -E 's/^code_gen_ipgen_//')
      component_xml="$abs/component.xml"
      cogen_root=$(echo "$abs" | grep -oE ".*code_gen_ipgen_[A-Za-z0-9_]+_hls_[0-9]+_[A-Za-z0-9]+" || true)
      if [ -n "$cogen_root" ] && [ -d "$cogen_root" ] && [ ! -d "$BACKUP_DIR/p${idx}_$(basename "$cogen_root")" ]; then
        cp -r "$cogen_root" "$BACKUP_DIR/p${idx}_$(basename "$cogen_root")" || echo "  WARN: backup failed for $cogen_root"
        echo "  backed up: $(basename "$cogen_root")"
      fi
      newname="p${idx}_${typename}"
      cat >> "$TCL_OUT" <<EOF
# partition $idx: $typename -> $newname
ipx::edit_ip_in_project -upgrade true -name tmp_edit_$newname -directory /tmp/ipedit_tiny_$newname {$component_xml}
set core [ipx::current_core]
set_property NAME $newname \$core
set_property VLNV [regsub {:[^:]+:[^:]+\$} [get_property VLNV \$core] ":$newname:1.0"] \$core
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
