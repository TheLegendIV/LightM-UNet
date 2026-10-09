open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files */top.bd]
reset_target all [get_files */top.bd]
generate_target all [get_files */top.bd]
report_ip_status -name u8in_regen_status
puts "=== REGEN_U8IN_DONE ==="
close_project
