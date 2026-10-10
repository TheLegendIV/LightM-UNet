open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files */top.bd]
set cell [get_bd_cells -hier -filter {NAME =~ *GenericPartition_1*  && TYPE == ip}]
puts "bd cells: $cell"
upgrade_ip [get_ips top_GenericPartition_1_0_0] -log /tmp/upgrade_p1.log
validate_bd_design -force
save_bd_design
generate_target all [get_files */top.bd] -force
set r [get_runs -quiet top_GenericPartition_1_0_0_synth_1]
if {$r ne ""} { reset_run $r; puts "reset $r" }
puts "=== report_ip_status ==="
report_ip_status
puts "=== UPGRADE_P1_DONE ==="
close_project
