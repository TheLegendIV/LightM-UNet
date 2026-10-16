# Forces a fresh synthesis of one swapped partition inside a block design: drops its stale OOC synth run, clears the project IP cache,
# recreates the missing OOC run from the current IP definition and resets the top-level runs. Other partitions' runs are left alone.
# (A BD-nested IP cannot be reset individually -- reset_target is refused -- so the run is deleted and recreated instead.)
# Usage: vivado -mode batch -source force_fresh_partition.tcl -tclargs <xpr> <partition_idx>
set XPR [lindex $argv 0]
set IDX [lindex $argv 1]
open_project $XPR
set run top_GenericPartition_${IDX}_0_0_synth_1

puts "=== clear project IP cache ==="
config_ip_cache -clear_output_repo

puts "=== runs before ==="
foreach r [get_runs] { puts "$r status=[get_property STATUS $r] needs_refresh=[get_property NEEDS_REFRESH $r]" }

if {[llength [get_runs -quiet $run]] > 0} {
    puts "=== deleting stale run $run ==="
    reset_run $run
    delete_runs $run
}
puts "=== creating missing OOC runs from the current IP definitions ==="
create_ip_run [get_files */top.bd]
reset_run synth_1
reset_run impl_1

puts "=== runs after ==="
foreach r [get_runs] { puts "$r status=[get_property STATUS $r] needs_refresh=[get_property NEEDS_REFRESH $r]" }
puts "=== old-pool sources left in the new run fileset (must be 0) ==="
puts [llength [get_files -quiet -of_objects [get_filesets $run] *StreamingMaxPool*]]
save_bd_design
puts "FRESH_DONE partition $IDX"
close_project
