open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set bd_file [get_files top.bd]
puts "re-regenerating output products for $bd_file (retry pass)"
generate_target {all} $bd_file -force

reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1

set status [get_property STATUS [get_runs synth_1]]
set progress [get_property PROGRESS [get_runs synth_1]]
puts "SYNTH_1_STATUS: $status ($progress)"

close_project
