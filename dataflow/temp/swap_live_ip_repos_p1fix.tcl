set NEW_BUILD /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_p1fix
set OLD_MARK /finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200/GenericPartition_1/

open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set existing [get_property ip_repo_paths [current_project]]
set kept {}
set dropped 0
foreach p $existing {
    if {[string first $OLD_MARK $p] >= 0} { incr dropped; puts "  DROP $p" } else { lappend kept $p }
}
puts "existing=[llength $existing] dropped_p1=$dropped kept=[llength $kept]"

set stitch [glob -nocomplain -type d $NEW_BUILD/GenericPartition_1/vivado_stitch_proj_*/ip]
if {[llength $stitch] != 1} { error "expected 1 stitched ip dir, got [llength $stitch]: $stitch" }
set hls [glob -nocomplain -type d $NEW_BUILD/GenericPartition_1/code_gen_ipgen_*_hls_*/*/sol1/impl/ip]
puts "  P1 stitch=1 hls_children=[llength $hls]"
set new_paths [concat $stitch $hls]

set all_paths [lsort -unique [concat $kept $new_paths]]
puts "new=[llength $new_paths] final=[llength $all_paths]"

set_property ip_repo_paths $all_paths [current_project]
update_ip_catalog -rebuild -scan_changes

puts "GenericPartition ipdefs: [llength [get_ipdefs -quiet *GenericPartition*]]"
foreach d [get_ipdefs -quiet *GenericPartition_1*] { puts "  $d [get_property REPOSITORY $d]" }

open_bd_design [get_files */top.bd]
puts "=== report_ip_status ==="
report_ip_status
puts "=== SWAP_P1FIX_DONE ==="
close_project
