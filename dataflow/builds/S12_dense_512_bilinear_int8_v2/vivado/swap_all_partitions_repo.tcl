# Re-points ALL GenericPartition IPs of an existing BD project at one FINN build and refreshes them (hand-edited BD parts -- clocking, latency monitor, IODMA -- are kept).
# Usage: vivado -mode batch -source swap_all_partitions_repo.tcl -tclargs <xpr> <new_build_dir>
#   every ip repo under finn_build_tmp/<any>/GenericPartition_<0-7>/ is dropped; the newest vivado_stitch_proj_*/ip of <new_build_dir>/GenericPartition_<i> + its HLS child repos are added.
# Then: upgrade the 8 IPs, validate, save, generate targets, and force a fresh synthesis state (drop the 8 stale OOC runs, clear the IP cache, recreate the runs, reset synth_1 / impl_1).
set XPR [lindex $argv 0]
set NEW [lindex $argv 1]
set EXTRA [lrange $argv 2 end]      ;# repos to keep / re-add regardless (the p0 / p7 IODMA HLS IPs live under an old build dir)

open_project $XPR
set keep {}
set dropped 0
foreach p [get_property ip_repo_paths [current_project]] {
    if {[regexp {finn_build_tmp/[^/]+/GenericPartition_[0-7]/} $p] && ![string match *IODMA* $p]} { incr dropped } else { lappend keep $p }
}
set added $EXTRA
for {set idx 0} {$idx < 8} {incr idx} {
    set pd $NEW/GenericPartition_$idx
    set stitch [lsort -decreasing -command {apply {{a b} {expr {[file mtime $a] - [file mtime $b]}}}} [glob -nocomplain -type d $pd/vivado_stitch_proj_*/ip]]
    if {[llength $stitch] == 0} { puts "ERROR: no stitched IP under $pd"; exit 1 }
    if {[llength $stitch] > 1} { puts "WARNING: [llength $stitch] stitched IP dirs under $pd, using the newest: [lindex $stitch 0]" }
    set n 1
    lappend added [lindex $stitch 0]
    foreach d [glob -nocomplain -type d $pd/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] { lappend added $d; incr n }
    puts "partition $idx: $n repos from $pd"
}
puts "=== dropped $dropped old partition repos, adding [llength $added] ==="
set_property ip_repo_paths [lsort -unique [concat $keep $added]] [current_project]
update_ip_catalog -rebuild -scan_changes

open_bd_design [get_files */top.bd]
for {set idx 0} {$idx < 8} {incr idx} {
    set cell [get_ips -quiet top_GenericPartition_${idx}_*]
    puts "=== upgrading $cell ==="
    if {[catch {upgrade_ip $cell} err]} { puts "UPGRADE_IP_ERROR partition $idx: $err" }
}
puts "=== validate_bd_design ==="
set vrc [catch {validate_bd_design -force} verr]
puts "VALIDATE_RESULT rc=$vrc $verr"
save_bd_design
puts "=== generate_target all ==="
set grc [catch {generate_target all [get_files */top.bd] -force} gerr]
puts "GENERATE_RESULT rc=$grc $gerr"
report_ip_status -name ip_status_after

puts "=== force fresh synthesis ==="
config_ip_cache -clear_output_repo
for {set idx 0} {$idx < 8} {incr idx} {
    set run top_GenericPartition_${idx}_0_0_synth_1
    if {[llength [get_runs -quiet $run]] > 0} { reset_run $run; delete_runs $run }
}
create_ip_run [get_files */top.bd]
reset_run synth_1
reset_run impl_1
foreach r [get_runs] { puts "RUN $r status=[get_property STATUS $r] needs_refresh=[get_property NEEDS_REFRESH $r]" }
puts "old-pool sources left in the partition 0 run fileset (must be 0): [llength [get_files -quiet -of_objects [get_filesets top_GenericPartition_0_0_synth_1] *StreamingMaxPool*]]"
save_bd_design
puts "SWAP_ALL_DONE"
close_project
