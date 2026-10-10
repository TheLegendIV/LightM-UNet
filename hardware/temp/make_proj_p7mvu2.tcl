set SRC_XPR  /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
set NEW_NAME s12_256_analytical_p7mvu2
set NEW_DIR  /home/thelegendiv/finn/vivado_projects/S12_256_analytical_p7mvu2/$NEW_NAME
set NEW_BUILD /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_p7mvu2
set OLD_MARK /finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_7/

open_project $SRC_XPR
save_project_as -force -exclude_run_results $NEW_NAME $NEW_DIR
puts "=== new project: [get_property DIRECTORY [current_project]] ==="

set existing [get_property ip_repo_paths [current_project]]
set kept {}
set dropped 0
foreach p $existing {
    if {[string first $OLD_MARK $p] >= 0} { incr dropped; puts "  DROP $p" } else { lappend kept $p }
}
puts "existing=[llength $existing] dropped_p7=$dropped kept=[llength $kept]"

set stitch [glob -nocomplain -type d $NEW_BUILD/GenericPartition_7/vivado_stitch_proj_*/ip]
if {[llength $stitch] != 1} { error "expected 1 stitched ip dir, got [llength $stitch]: $stitch" }
set hls [glob -nocomplain -type d $NEW_BUILD/GenericPartition_7/code_gen_ipgen_*_hls_*/*/sol1/impl/ip]
puts "  P7 stitch=1 hls_children=[llength $hls]"
set new_paths [concat $stitch $hls]

set all_paths [lsort -unique [concat $kept $new_paths]]
puts "new=[llength $new_paths] final=[llength $all_paths]"

set_property ip_repo_paths $all_paths [current_project]
update_ip_catalog -rebuild -scan_changes

foreach d [get_ipdefs -quiet *GenericPartition_7*] { puts "  $d [get_property REPOSITORY $d]" }

open_bd_design [get_files */top.bd]
upgrade_ip [get_ips top_GenericPartition_7_0_0] -log /tmp/upgrade_p7mvu2.log
validate_bd_design -force
save_bd_design
generate_target all [get_files */top.bd] -force
puts "=== report_ip_status ==="
report_ip_status
puts "=== runs ==="
foreach r [get_runs] { puts "$r [get_property STATUS $r]" }
puts "=== MAKE_PROJ_P7MVU2_DONE ==="
close_project
