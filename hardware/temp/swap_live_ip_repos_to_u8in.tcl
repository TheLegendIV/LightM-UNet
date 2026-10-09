set NEW_BUILD /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_u8in_int6_fps250_lat200
set OLD_MARK /finn_build_tmp/S12_dense_256_u4_analytical_v2_ft15ep_int6_fps250_lat200/
set FINN_ROOT /home/thelegendiv/finn

open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set existing [get_property ip_repo_paths [current_project]]
set kept {}
set dropped 0
foreach p $existing {
    if {[string first $OLD_MARK $p] >= 0} { incr dropped } else { lappend kept $p }
}
puts "existing=[llength $existing] dropped_v2=$dropped kept=[llength $kept]"
foreach p $kept { puts "  KEEP $p" }

set new_paths {}
foreach i {0 1 2 3 4 5 6 7} {
    set stitch [glob -nocomplain -type d $NEW_BUILD/GenericPartition_$i/vivado_stitch_proj_*/ip]
    if {[llength $stitch] != 1} { error "partition $i: expected 1 stitched ip dir, got [llength $stitch]: $stitch" }
    set hls [glob -nocomplain -type d $NEW_BUILD/GenericPartition_$i/code_gen_ipgen_*_hls_*/*/sol1/impl/ip]
    puts "  P$i stitch=1 hls_children=[llength $hls]"
    lappend new_paths {*}$stitch {*}$hls
}
lappend new_paths $FINN_ROOT/finn-rtllib/memstream

set all_paths [lsort -unique [concat $kept $new_paths]]
puts "new=[llength $new_paths] final=[llength $all_paths]"

set_property ip_repo_paths $all_paths [current_project]
update_ip_catalog -rebuild -scan_changes

puts "=== catalog check ==="
puts "GenericPartition ipdefs: [llength [get_ipdefs -quiet *GenericPartition*]]"
puts "memstream ipdefs: [llength [get_ipdefs -quiet *memstream*]]"
puts "IODMA ipdefs: [llength [get_ipdefs -quiet *IODMA*]]"

open_bd_design [get_files */top.bd]
puts "=== report_ip_status ==="
report_ip_status
puts "=== SWAP_U8IN_DONE ==="
close_project
