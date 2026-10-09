open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files */top.bd]
set cells [get_ips top_GenericPartition_*]
puts "upgrading [llength $cells] ips"
upgrade_ip -vlnv {} $cells
report_ip_status -name u8in_status
validate_bd_design
save_bd_design
puts "=== STATUS ==="
foreach ip [get_ips top_GenericPartition_*] { puts "[get_property NAME $ip] upgrade=[get_property UPGRADE_VERSIONS $ip] locked=[get_property IS_LOCKED $ip]" }
puts "=== UPGRADE_U8IN_DONE ==="
close_project
