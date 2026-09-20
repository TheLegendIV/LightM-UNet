set p0_base /tmp/finn_dev_thelegendiv/vivado_stitch_proj_hf0tokm3
set p1_base /tmp/finn_dev_thelegendiv/vivado_stitch_proj_rtj421tm

create_project -force tiny_combine_test2 /tmp/finn_dev_thelegendiv/tiny_combine_test2 -part xczu7ev-ffvc1156-2-e
set_property ip_repo_paths [list \
  $p0_base/ip \
  $p1_base/ip \
  /home/thelegendiv/finn/finn-rtllib/memstream \
  $p0_base/../code_gen_ipgen_DuplicateStreams_hls_0_9imqhfgr/project_DuplicateStreams_hls_0/sol1/impl/ip \
  $p0_base/../code_gen_ipgen_AddStreams_hls_0_7rr8i5jm/project_AddStreams_hls_0/sol1/impl/ip \
  $p0_base/../code_gen_ipgen_MVAU_hls_0_h6s8cvt4/project_MVAU_hls_0/sol1/impl/ip \
  $p1_base/../code_gen_ipgen_DuplicateStreams_hls_0_2pc4w53x/project_DuplicateStreams_hls_0/sol1/impl/ip \
  $p1_base/../code_gen_ipgen_AddStreams_hls_0_j8s__otq/project_AddStreams_hls_0/sol1/impl/ip \
  $p1_base/../code_gen_ipgen_MVAU_hls_0_bez4stje/project_MVAU_hls_0/sol1/impl/ip \
  $p0_base/../code_gen_ipgen_StreamingFIFO_rtl_0_9u6w_e6m \
  $p1_base/../code_gen_ipgen_StreamingFIFO_rtl_0_0ppp80qt \
  $p0_base/../code_gen_ipgen_MVAU_rtl_0_eo10d7_d \
  $p1_base/../code_gen_ipgen_MVAU_rtl_0_njzdpdz9 \
  $p0_base/../code_gen_ipgen_Thresholding_rtl_0_ms6jemgw \
  $p1_base/../code_gen_ipgen_Thresholding_rtl_0_iesb_j5y \
  $p0_base/../code_gen_ipgen_StreamingDataWidthConverter_rtl_0_4_6pkjvt \
  $p1_base/../code_gen_ipgen_StreamingDataWidthConverter_rtl_0_bi3cpcj5 \
  $p0_base/../code_gen_ipgen_FMPadding_rtl_0_vi677tqv \
  $p1_base/../code_gen_ipgen_FMPadding_rtl_0_9rpubdhj \
  $p0_base/../code_gen_ipgen_ConvolutionInputGenerator_rtl_0_lzyhyyg1 \
  $p1_base/../code_gen_ipgen_ConvolutionInputGenerator_rtl_0_fwb5pgpk \
] [current_project]
update_ip_catalog -rebuild

puts "===CATALOG_VLNV_COUNTS==="
set all_ips [get_ipdefs -all]
foreach vlnv {xilinx.com:hls:DuplicateStreams_hls_0:1.0 xilinx.com:hls:AddStreams_hls_0:1.0 xilinx.com:hls:MVAU_hls_0:1.0 xilinx.com:hls:StreamingFIFO_rtl_0:1.0 xilinx.com:hls:MVAU_rtl_0:1.0 xilinx.com:hls:Thresholding_rtl_0:1.0 xilinx.com:hls:StreamingDataWidthConverter_rtl_0:1.0 xilinx.com:hls:FMPadding_rtl_0:1.0 xilinx.com:hls:ConvolutionInputGenerator_rtl_0:1.0 xilinx.com:hls:p0_DuplicateStreams_hls_0:1.0 xilinx.com:hls:p1_DuplicateStreams_hls_0:1.0} {
  set matches [get_ipdefs -all -vlnv $vlnv]
  puts "VLNV $vlnv -> [llength $matches] catalog match(es): $matches"
}
puts "===END_CATALOG_VLNV_COUNTS==="

create_bd_design "top_combine2"
create_bd_port -dir I -type clk ap_clk
create_bd_port -dir I -type rst ap_rst_n
set p0 [create_bd_cell -type ip -vlnv xilinx_finn:finn:tiny_p0_finn_design:1.0 p0_finn_design]
set p1 [create_bd_cell -type ip -vlnv xilinx_finn:finn:tiny_p1_finn_design:1.0 p1_finn_design]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins p0_finn_design/ap_clk]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins p1_finn_design/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins p0_finn_design/ap_rst_n]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins p1_finn_design/ap_rst_n]

puts "===IP_STATUS_LIST==="
foreach ip [get_ips -all] {
  puts "IP: $ip  VLNV: [get_property IPDEF $ip]  LOCKED: [get_property IS_LOCKED $ip]"
}
puts "===END_IP_STATUS_LIST==="

if {[catch {validate_bd_design -force} errmsg]} {
  puts "===VALIDATE_FAILED=== $errmsg"
} else {
  puts "===VALIDATE_OK==="
}

save_bd_design
close_project
