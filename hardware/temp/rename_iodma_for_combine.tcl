create_project -force scratch_rename_iodma_proj /tmp/scratch_rename_iodma_proj -part xczu7ev-ffvc1156-2-e

# idma from zynqp0 build: StreamingDataflowPartition_0_IODMA_hls_0 -> zynqp0_idma_IODMA_hls_0
ipx::edit_ip_in_project -upgrade true -name tmp_edit_zynqp0_idma -directory /tmp/ipedit_zynqp0_idma {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/code_gen_ipgen_StreamingDataflowPartition_0_IODMA_hls_0_66qk80st/project_StreamingDataflowPartition_0_IODMA_hls_0/sol1/impl/ip/component.xml}
set core [ipx::current_core]
set_property NAME zynqp0_idma_IODMA_hls_0 $core
set_property VLNV [regsub {:[^:]+:[^:]+$} [get_property VLNV $core] ":zynqp0_idma_IODMA_hls_0:1.0"] $core
set_property value {zynqp0_idma_IODMA_hls_0_v1_0} [ipx::get_user_parameters Component_Name -of_objects $core]
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project -delete

# odma from zynqp7 build: StreamingDataflowPartition_2_IODMA_hls_0 -> zynqp7_odma_IODMA_hls_0
ipx::edit_ip_in_project -upgrade true -name tmp_edit_zynqp7_odma -directory /tmp/ipedit_zynqp7_odma {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/zynqbuild_partition7/code_gen_ipgen_StreamingDataflowPartition_2_IODMA_hls_0_pu7nilya/project_StreamingDataflowPartition_2_IODMA_hls_0/sol1/impl/ip/component.xml}
set core [ipx::current_core]
set_property NAME zynqp7_odma_IODMA_hls_0 $core
set_property VLNV [regsub {:[^:]+:[^:]+$} [get_property VLNV $core] ":zynqp7_odma_IODMA_hls_0:1.0"] $core
set_property value {zynqp7_odma_IODMA_hls_0_v1_0} [ipx::get_user_parameters Component_Name -of_objects $core]
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project -delete

close_project
