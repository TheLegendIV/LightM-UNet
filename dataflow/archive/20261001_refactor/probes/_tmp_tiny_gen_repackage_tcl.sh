#!/bin/bash
# Adapted from hardware/temp/_tmp_gen_repackage_tcl.sh for the tiny 2-partition test.
# Distinct -module name per partition (tiny_p{idx}_finn_design) to avoid a
# TOP-LEVEL stitched-IP name/VLNV collision too, since both naive builds used
# the default ip_name="finn_design".
BASE=/tmp/finn_dev_thelegendiv
OUT_DIR="$BASE/tiny_repackage_tcl"
mkdir -p "$OUT_DIR"

declare -A DIRMAP=(
  [0]=vivado_stitch_proj_hf0tokm3
  [1]=vivado_stitch_proj_rtj421tm
)

for idx in 0 1; do
  d="${DIRMAP[$idx]}"
  proj_dir="$BASE/$d"
  bdname="finn_design"
  module_name="tiny_p${idx}_finn_design"
  xpr="$proj_dir/finn_vivado_stitch_proj.xpr"
  out="$OUT_DIR/repackage_partition_${idx}.tcl"

  {
    echo "open_project {$xpr}"
    echo ""
    echo "update_ip_catalog -rebuild"
    echo "open_bd_design [get_files ${bdname}.bd]"
    echo "report_ip_status -name ip_status"
    echo "upgrade_ip [get_ips -all]"
    echo "validate_bd_design -force"
    echo "save_bd_design"
    echo "generate_target all [get_files ${bdname}.bd] -force"
    echo ""
    echo "ipx::package_project -root_dir $proj_dir/ip -vendor xilinx_finn -library finn -taxonomy /UserIP -module $bdname -import_files -force"
    echo "set core [ipx::current_core]"
    echo "set_property NAME $module_name \$core"
    echo "set_property VLNV [regsub {:[^:]+:[^:]+\$} [get_property VLNV \$core] \":$module_name:1.0\"] \$core"
    echo "set_property value {${module_name}_v1_0} [ipx::get_user_parameters Component_Name -of_objects \$core]"
    echo "set_property core_revision 2 \$core"
    echo "ipx::create_xgui_files \$core"
    echo "ipx::update_checksums \$core"
    echo "ipx::save_core \$core"
    echo "close_project"
  } > "$out"
  echo "wrote $out"
done
