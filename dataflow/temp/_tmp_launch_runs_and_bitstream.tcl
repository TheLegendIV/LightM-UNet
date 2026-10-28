open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set partition_runs [get_runs -filter {NAME =~ "*GenericPartition*_synth_1"}]
puts "Launching [llength $partition_runs] partition synth runs: $partition_runs"
launch_runs $partition_runs -jobs 12
wait_on_runs $partition_runs

set failed 0
foreach r $partition_runs {
    set status [get_property STATUS $r]
    set progress [get_property PROGRESS $r]
    puts "$r : $status ($progress)"
    if {![string match "*Complete!*" $status]} {
        puts "ERROR: $r did not complete successfully"
        set failed 1
    }
}

if {$failed} {
    puts "ABORTING: one or more partition synth runs failed, not launching top-level impl."
    close_project
    exit 1
}

# top-level synth_1/impl_1 must pick up the newly re-synthesized partition checkpoints
reset_run synth_1
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 12
wait_on_runs impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "impl_1 status: $impl_status"
if {![string match "*write_bitstream Complete!*" $impl_status]} {
    puts "ERROR: impl_1 did not reach write_bitstream completion"
    close_project
    exit 1
}

report_utilization -file /tmp/final_util_after_directives.rpt
puts "Bitstream generation complete."

close_project
