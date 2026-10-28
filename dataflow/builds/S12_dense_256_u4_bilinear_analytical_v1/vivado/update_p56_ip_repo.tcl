# Repoint the bilinear 256 project's IP repos at the rebuilt P5/P6 IPs (replacing the pre-fix ones, same VLNV) and upgrade those two cells.
set XPR    [lindex $argv 0]
set BUILD  [lindex $argv 1]
set CUTOFF [clock scan {2026-10-10 12:00:00}]

open_project $XPR
set old_paths [get_property ip_repo_paths [current_project]]
puts "=== [llength $old_paths] existing ip_repo_paths ==="

set cand {}
foreach p {5 6} {
    set pd $BUILD/GenericPartition_$p
    foreach d [glob -nocomplain -type d $pd/vivado_stitch_proj_*/ip] { lappend cand $d }
    foreach d [glob -nocomplain -type d $pd/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] { lappend cand $d }
}
set new_p56 {}
foreach d $cand { if {[file mtime $d] > $CUTOFF} { lappend new_p56 $d } }
puts "=== [llength $new_p56] rebuilt P5/P6 ip dirs ==="

set keep {}
set dropped 0
foreach d $old_paths {
    if {[regexp {GenericPartition_[56]/(vivado_stitch_proj_|code_gen_ipgen_)} $d] && [lsearch -exact $new_p56 [file normalize $d]] < 0} {
        incr dropped
        puts "DROP $d"
    } else { lappend keep $d }
}
set added 0
foreach d $new_p56 {
    if {[lsearch -exact $keep $d] < 0} { lappend keep $d; incr added }
}
puts "=== dropped $dropped stale, added $added new ==="
set_property ip_repo_paths [lsort -unique $keep] [current_project]
update_ip_catalog -rebuild -scan_changes

open_bd_design [get_files */top.bd]
report_ip_status -name ip_status_before
set cells [get_ips -quiet {top_GenericPartition_5_* top_GenericPartition_6_*}]
puts "=== upgrading [llength $cells] cells: $cells ==="
if {[catch {upgrade_ip $cells} err]} { puts "UPGRADE_IP_ERROR: $err" }
set vrc [catch {validate_bd_design -force} verr]
puts "VALIDATE_RESULT rc=$vrc $verr"
save_bd_design
set grc [catch {generate_target all [get_files */top.bd] -force} gerr]
puts "GENERATE_RESULT rc=$grc $gerr"
report_ip_status -name ip_status_after
puts "REPO_UPDATE_DONE"
close_project
