ipx::edit_ip_in_project -upgrade true -name edit_ip_tmp_idma -directory /tmp/edit_ip_tmp_idma \
    /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_ydzflm07/ip/component.xml
set core [ipx::current_core]
set_property NAME StreamingDataflowPartition_0_zynqp0_idma $core
set_property VLNV xilinx_finn:finn:StreamingDataflowPartition_0_zynqp0_idma:1.0 $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project -delete

ipx::edit_ip_in_project -upgrade true -name edit_ip_tmp_kernel -directory /tmp/edit_ip_tmp_kernel \
    /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_stitch_proj_4c00t1f9/ip/component.xml
set core [ipx::current_core]
set_property NAME StreamingDataflowPartition_1_zynqp0_kernel $core
set_property VLNV xilinx_finn:finn:StreamingDataflowPartition_1_zynqp0_kernel:1.0 $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project -delete
