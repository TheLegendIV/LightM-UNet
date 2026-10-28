# Duplicates a hand-edited BD project (PS clocking, latency monitor, address map, IODMA IPs all kept) under a new name and points
# its FINN partition IP repos at the new build(s). Everything that is not a GenericPartition repo of the base build is kept.
# Usage: vivado -mode batch -source make_project_from_base.tcl -tclargs <base_xpr> <new_proj_name> <new_proj_dir> \
#           <old_build_dir_substring> <default_build_dir> [<alt_build_dir> <alt_partition_ids_tcl_list>]
# <old_build_dir_substring>: repos whose path contains "<it>/GenericPartition_" are dropped (the base build's partitions).
# Partition i is taken from <alt_build_dir> if i is in <alt_partition_ids>, otherwise from <default_build_dir>.

set BASE_XPR   [lindex $argv 0]
set NAME       [lindex $argv 1]
set DIR        [lindex $argv 2]
set OLD_SUB    [lindex $argv 3]
set BUILD_MAIN [lindex $argv 4]
set BUILD_ALT  [expr {[llength $argv] >= 7 ? [lindex $argv 5] : ""}]
set ALT_IDS    [expr {[llength $argv] >= 7 ? [lindex $argv 6] : {}}]

open_project -read_only $BASE_XPR
save_project_as -force -exclude_run_results $NAME $DIR

set keep {}
set dropped 0
foreach p [get_property ip_repo_paths [current_project]] {
    if {[string match "*${OLD_SUB}/GenericPartition_*" $p]} { incr dropped } else { lappend keep $p }
}
puts "=== kept [llength $keep] ip repos, dropped $dropped base-build partition repos ==="

set added {}
for {set i 0} {$i < 8} {incr i} {
    set b [expr {[lsearch -exact $ALT_IDS $i] >= 0 ? $BUILD_ALT : $BUILD_MAIN}]
    set pd $b/GenericPartition_$i
    set stitch [lsort -decreasing -command {apply {{a b} {expr {[file mtime $a] - [file mtime $b]}}}} [glob -nocomplain -type d $pd/vivado_stitch_proj_*/ip]]
    if {[llength $stitch] == 0} { puts "ERROR: no stitched IP under $pd"; exit 1 }
    if {[llength $stitch] > 1} { puts "WARNING: partition $i has [llength $stitch] stitched IP dirs, using the newest: [lindex $stitch 0]" }
    lappend added [lindex $stitch 0]
    foreach d [glob -nocomplain -type d $pd/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] { lappend added $d }
    puts "partition $i: [lindex $stitch 0]  (from $b)"
}
set ip_paths [lsort -unique [concat $keep $added]]
puts "=== [llength $ip_paths] ip_repo_paths ==="
set_property ip_repo_paths $ip_paths [current_project]
update_ip_catalog -rebuild -scan_changes

open_bd_design [get_files */top.bd]
report_ip_status -name ip_status_before

set finn_ips [get_ips -quiet {top_GenericPartition_*}]
puts "=== upgrading [llength $finn_ips] FINN partition IPs ==="
if {[catch {upgrade_ip $finn_ips} err]} { puts "UPGRADE_IP_ERROR: $err" }

puts "=== validate_bd_design ==="
set vrc [catch {validate_bd_design -force} verr]
puts "VALIDATE_RESULT rc=$vrc $verr"
save_bd_design

puts "=== generate_target all ==="
set grc [catch {generate_target all [get_files */top.bd] -force} gerr]
puts "GENERATE_RESULT rc=$grc $gerr"
report_ip_status -name ip_status_after
puts "MAKE_PROJECT_DONE $DIR"
close_project
