# Derives the 512 BD project from the 256 reference project: same block design
# (PS, interconnect, ILA, 8 GenericPartition cells, p0_input/p7_output IODMA),
# with all FINN IP repos pointed at the 512 build.
# Usage: vivado -mode batch -source make_project_512.tcl -tclargs \
#           <base_xpr> <new_proj_name> <new_proj_dir> <finn_build_dir> [finn_root] [extra_ip_repo_dirs_tcl_list]

set BASE_XPR [lindex $argv 0]
set NAME     [lindex $argv 1]
set DIR      [lindex $argv 2]
set BUILD    [lindex $argv 3]
set FINN_ROOT [expr {[llength $argv] >= 5 ? [lindex $argv 4] : "/home/thelegendiv/finn"}]
set EXTRA_REPOS [expr {[llength $argv] >= 6 ? [lindex $argv 5] : {}}]

open_project $BASE_XPR
save_project_as -force -exclude_run_results $NAME $DIR

set ip_paths {}
foreach part_dir [lsort [glob -nocomplain -type d $BUILD/GenericPartition_*]] {
    foreach d [glob -nocomplain -type d $part_dir/vivado_stitch_proj_*/ip] { lappend ip_paths $d }
    foreach d [glob -nocomplain -type d $part_dir/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] { lappend ip_paths $d }
}
lappend ip_paths $FINN_ROOT/finn-rtllib/memstream
foreach d $EXTRA_REPOS { lappend ip_paths $d }
set ip_paths [lsort -unique $ip_paths]
puts "=== [llength $ip_paths] ip_repo_paths ==="
set_property ip_repo_paths $ip_paths [current_project]
update_ip_catalog -rebuild -scan_changes

open_bd_design [get_files */top.bd]
report_ip_status -name ip_status_before

set finn_ips [get_ips -quiet {top_GenericPartition_* top_p0_input_IODMA_* top_p7_output_IODMA_*}]
puts "=== upgrading [llength $finn_ips] FINN IPs ==="
if {[catch {upgrade_ip $finn_ips} err]} { puts "UPGRADE_IP_ERROR: $err" }

puts "=== validate_bd_design ==="
set vrc [catch {validate_bd_design -force} verr]
puts "VALIDATE_RESULT rc=$vrc $verr"
save_bd_design

puts "=== generate_target all ==="
set grc [catch {generate_target all [get_files */top.bd] -force} gerr]
puts "GENERATE_RESULT rc=$grc $gerr"
report_ip_status -name ip_status_after
puts "MAKE_PROJECT_512_DONE $DIR"
close_project
