open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_6_75nd4k/finn_vivado_stitch_proj.xpr}
update_ip_catalog -rebuild
open_bd_design [get_files StreamingDataflowPartition_0.bd]
foreach c [get_bd_cells -hierarchical] {
  puts "CELL: $c  VLNV: [get_property VLNV $c]  REF_NAME: [get_property REF_NAME $c]  IS_LOCKED: [get_property IS_LOCKED $c]"
}
close_project
