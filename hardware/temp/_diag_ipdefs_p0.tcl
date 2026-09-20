open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_6_75nd4k/finn_vivado_stitch_proj.xpr}
update_ip_catalog -rebuild
puts "IPDEFS_MATCHING:"
puts [get_ipdefs -all -filter {NAME =~ "*DuplicateStreams*"}]
puts "---"
puts [get_ipdefs -all -vlnv xilinx.com:hls:p0_DuplicateStreams_hls_0:1.0]
close_project
