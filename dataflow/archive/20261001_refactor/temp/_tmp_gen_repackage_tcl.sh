#!/bin/bash
# For each partition: open its own stitched-IP project, apply that
# partition's slice of the child-IP renames, refresh the catalog, validate +
# regenerate the (already-patched) BD, and re-package the partition's own
# top-level stitched IP so its component.xml/checksums reflect the new
# child VLNVs. Run non-interactively per partition via:
#   vivado -mode batch -source repackage_partition_N.tcl -log ... -journal ...
#
# IMPORTANT: run _tmp_gen_rename_catalog_ips.sh and _tmp_patch_partition_bds.sh
# FIRST (both already done). This script only GENERATES the per-partition
# Tcl files -- it does not invoke Vivado itself.
BASE=/home/thelegendiv/finn/notebooks/enet/finn_build_tmp
RENAME_TCL="$BASE/rename_catalog_ips_for_combine.tcl"
OUT_DIR="$BASE/repackage_tcl"
mkdir -p "$OUT_DIR"

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
  xpr="$proj_dir/finn_vivado_stitch_proj.xpr"
  out="$OUT_DIR/repackage_partition_${idx}.tcl"

  {
    echo "# Auto-generated: repackage partition $idx ($d) after catalog IP rename"
    echo "# NOTE: run rename_catalog_ips_for_combine.tcl FIRST, standalone, with NO"
    echo "# project open (ipx::edit_ip_in_project manages its own temp project and"
    echo "# will conflict with an already-open partition project otherwise)."
    echo "open_project {$xpr}"
    echo ""
    echo "# --- refresh catalog (picks up the renamed catalog child IPs) ---"
    echo "update_ip_catalog -rebuild"
    echo "open_bd_design [get_files ${bdname}.bd]"
    echo "report_ip_status -name ip_status"
    echo "upgrade_ip [get_ips -all]"
    echo "validate_bd_design -force"
    echo "save_bd_design"
    echo "generate_target all [get_files ${bdname}.bd] -force"
    echo ""
    echo "# --- re-package this partition's own top-level stitched IP ---"
    echo "ipx::package_project -root_dir $proj_dir/ip -vendor xilinx_finn -library finn -taxonomy /UserIP -module $bdname -import_files -force"
    echo "set core [ipx::current_core]"
    echo "set_property core_revision 2 \$core"
    echo "ipx::create_xgui_files \$core"
    echo "ipx::update_checksums \$core"
    echo "ipx::save_core \$core"
    echo "close_project"
  } > "$out"
  echo "wrote $out"
done
