create_project -force tmp_q /tmp/tmp_q -part xczu7ev-ffvc1156-2-e
ipx::open_ipxact_file /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip/component.xml
puts "CMDS: [lsort [info commands ipx::*view*]] | [lsort [info commands ipx::*model*]] | [lsort [info commands ipx::*file_group*]]"
set core [ipx::current_core]
puts "CORE: $core"
puts "FG: [ipx::get_file_groups -of_objects $core]"
foreach fg [ipx::get_file_groups -of_objects $core] { puts "FGPROP [get_property NAME $fg] [get_property MODEL_NAME $fg]" }
