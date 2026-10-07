set old_iodma_paths [list \
  "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip" \
  "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip" \
]

open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set existing [get_property ip_repo_paths [current_project]]
puts "Existing ip_repo_paths count: [llength $existing]"

set all_paths $existing
foreach p $old_iodma_paths {
    if {[lsearch -exact $all_paths $p] == -1} {
        lappend all_paths $p
    }
}

set_property ip_repo_paths $all_paths [current_project]
update_ip_catalog -rebuild -scan_changes

puts "Final ip_repo_paths ([llength $all_paths] entries):"
foreach p $all_paths { puts "  $p" }

close_project
