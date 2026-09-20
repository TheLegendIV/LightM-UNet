create_project -force tiny_combine_test /tmp/finn_dev_thelegendiv/tiny_combine_test -part xczu7ev-ffvc1156-2-e
set_property ip_repo_paths {
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_hf0tokm3/ip
  /tmp/finn_dev_thelegendiv/vivado_stitch_proj_rtj421tm/ip
} [current_project]
update_ip_catalog -rebuild

create_bd_design "top_combine"
set p0 [create_bd_cell -type ip -vlnv xilinx_finn:finn:tiny_p0_finn_design:1.0 p0_finn_design]
set p1 [create_bd_cell -type ip -vlnv xilinx_finn:finn:tiny_p1_finn_design:1.0 p1_finn_design]

report_ip_status -name combine_ip_status
puts "===IP_STATUS_LIST==="
foreach ip [get_ips -all] {
  puts "IP: $ip  VLNV: [get_property IPDEF $ip]  STATUS: [get_property IS_LOCKED $ip]"
}
puts "===END_IP_STATUS_LIST==="

validate_bd_design -force
puts "===VALIDATE_DONE==="

save_bd_design
close_project
