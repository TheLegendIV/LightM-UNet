open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_6_75nd4k/finn_vivado_stitch_proj.xpr}
update_ip_catalog -rebuild
open_bd_design [get_files StreamingDataflowPartition_0.bd]
set c [get_bd_cells /DuplicateStreams_hls_0]
puts "CELL=$c"
set ipobj [get_ips -of_objects $c]
puts "IPOBJ=$ipobj"
if {[llength $ipobj] > 0} {
  puts "IPOBJ_VLNV=[get_property IPDEF $ipobj]"
  puts "IPOBJ_UPGRADE_VERSIONS=[get_property UPGRADE_VERSIONS $ipobj]"
}
puts "CELL_VLNV=[get_property VLNV $c]"
puts "CELL_CONFIG_VLNV=[get_property CONFIG.Component_Name $c]"
close_project
