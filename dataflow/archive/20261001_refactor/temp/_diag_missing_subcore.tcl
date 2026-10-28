open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
puts "IP_REPO_PATHS = [get_property ip_repo_paths [current_project]]"
puts "--- ip catalog search for p0_DuplicateStreams_hls_0 ---"
puts [get_ipdefs -all *p0_DuplicateStreams_hls_0*]
puts "--- ip catalog search for p2_AddStreams_hls_0 ---"
puts [get_ipdefs -all *p2_AddStreams_hls_0*]
close_project
