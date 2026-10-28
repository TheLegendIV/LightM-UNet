# Re-points ONE partition of an existing BD project at a different FINN build and refreshes its IP.
# Usage: vivado -mode batch -source swap_partition_repo.tcl -tclargs <xpr> <partition_idx> <old_build_dir_substring> <new_build_dir>
#   <old_build_dir_substring>: ip repos containing "<it>/GenericPartition_<idx>/" are dropped
#   <new_build_dir>: dir containing GenericPartition_<idx> (newest vivado_stitch_proj_*/ip + its HLS child ip repos are added)
set XPR     [lindex $argv 0]
set IDX     [lindex $argv 1]
set OLD_SUB [lindex $argv 2]
set NEW     [lindex $argv 3]

open_project $XPR
set keep {}
set dropped 0
foreach p [get_property ip_repo_paths [current_project]] {
    if {[string match "*${OLD_SUB}/GenericPartition_${IDX}/*" $p]} { incr dropped } else { lappend keep $p }
}
set pd $NEW/GenericPartition_$IDX
set stitch [lsort -decreasing -command {apply {{a b} {expr {[file mtime $a] - [file mtime $b]}}}} [glob -nocomplain -type d $pd/vivado_stitch_proj_*/ip]]
if {[llength $stitch] == 0} { puts "ERROR: no stitched IP under $pd"; exit 1 }
if {[llength $stitch] > 1} { puts "WARNING: [llength $stitch] stitched IP dirs under $pd, using the newest: [lindex $stitch 0]" }
set added [list [lindex $stitch 0]]
foreach d [glob -nocomplain -type d $pd/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] { lappend added $d }
puts "=== partition $IDX: dropped $dropped repos, adding [llength $added] from $pd ==="
set_property ip_repo_paths [lsort -unique [concat $keep $added]] [current_project]
update_ip_catalog -rebuild -scan_changes

open_bd_design [get_files */top.bd]
set cell [get_ips -quiet top_GenericPartition_${IDX}_*]
puts "=== upgrading $cell ==="
if {[catch {upgrade_ip $cell} err]} { puts "UPGRADE_IP_ERROR: $err" }

puts "=== validate_bd_design ==="
set vrc [catch {validate_bd_design -force} verr]
puts "VALIDATE_RESULT rc=$vrc $verr"
save_bd_design
puts "=== generate_target all ==="
set grc [catch {generate_target all [get_files */top.bd] -force} gerr]
puts "GENERATE_RESULT rc=$grc $gerr"
report_ip_status -name ip_status_after
puts "SWAP_DONE partition $IDX"
close_project
