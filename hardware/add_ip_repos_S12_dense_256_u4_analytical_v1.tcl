# Registers all per-partition stitched IPs + both IODMA cores as IP
# repositories in the currently open Vivado project (run after `open_project`
# / `create_project`, before adding them to a block design).
# Build: S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200_milpfold_8way_20261005_235400

set BASE "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200"

set ip_repo_dirs [list \
    "$BASE/GenericPartition_0/vivado_stitch_proj_79kwx99z/ip" \
    "$BASE/GenericPartition_1/vivado_stitch_proj_0r930haf/ip" \
    "$BASE/GenericPartition_2/vivado_stitch_proj_097uavns/ip" \
    "$BASE/GenericPartition_3/vivado_stitch_proj_45d5_uvz/ip" \
    "$BASE/GenericPartition_4/vivado_stitch_proj_zmy9mpm5/ip" \
    "$BASE/GenericPartition_5/vivado_stitch_proj_7jkjhw4h/ip" \
    "$BASE/GenericPartition_6/vivado_stitch_proj_mcf4k7vm/ip" \
    "$BASE/GenericPartition_7/vivado_stitch_proj_8crqly7i/ip" \
    "$BASE/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23" \
    "$BASE/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g" \
]

foreach d $ip_repo_dirs {
    if {![file isdirectory $d]} {
        puts "WARNING: IP repo dir not found, skipping: $d"
    }
}

set_property ip_repo_paths $ip_repo_dirs [current_project]
update_ip_catalog -rebuild

puts "Registered [llength $ip_repo_dirs] IP repositories."
