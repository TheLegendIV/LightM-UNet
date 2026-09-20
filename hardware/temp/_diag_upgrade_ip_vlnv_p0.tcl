open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_6_75nd4k/finn_vivado_stitch_proj.xpr}
update_ip_catalog -rebuild
open_bd_design [get_files StreamingDataflowPartition_0.bd]
upgrade_ip -vlnv xilinx.com:hls:p0_DuplicateStreams_hls_0:1.0 [get_bd_cells /DuplicateStreams_hls_0]
upgrade_ip -vlnv xilinx.com:hls:p0_StreamingConcat_hls_0:1.0 [get_bd_cells /StreamingConcat_hls_0]
upgrade_ip -vlnv xilinx.com:hls:p0_StreamingMaxPool_hls_0:1.0 [get_bd_cells /StreamingMaxPool_hls_0]
validate_bd_design -force
puts "VALIDATE_OK"
close_project
