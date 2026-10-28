open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

set finn_build /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200

# 1) each partition's own packaged top-level ip/ dir
set ip_paths {}
foreach i {0 1 2 3 4 5 6 7} {
    foreach d [glob -nocomplain $finn_build/GenericPartition_$i/vivado_stitch_proj_*/ip] {
        lappend ip_paths $d
    }
}
# 2) every HLS-backend child node's own standalone exported ip dir, per partition
foreach i {0 1 2 3 4 5 6 7} {
    foreach d [glob -nocomplain $finn_build/GenericPartition_$i/code_gen_ipgen_*_hls_*/*/sol1/impl/ip] {
        lappend ip_paths $d
    }
}
# 3) FINN's own custom weight-memory IP (shared, single entry)
lappend ip_paths /home/thelegendiv/finn/finn-rtllib/memstream
set ip_paths [lsort -unique $ip_paths]

puts "=== setting [llength $ip_paths] ip_repo_paths entries ==="
set_property ip_repo_paths $ip_paths [current_project]
update_ip_catalog -rebuild -scan_changes

open_bd_design [get_files */top.bd]
puts "=== validating bd design ==="
validate_bd_design -force

puts "=== generate_target all ==="
generate_target all [get_files */top.bd] -force

puts "=== LIVE_PROJECT_FIX_COMPLETE_NO_FATAL_ERROR ==="
