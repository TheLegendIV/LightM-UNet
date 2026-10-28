# Fixes VLNV/name collision between the input and output IODMA_hls IP cores
# (both generated as node name "IODMA_hls_0" -> identical VLNV by default).
# Run this ONCE, before sourcing add_ip_repos_S12_dense_256_u4_analytical_v1.tcl
# (it edits the component.xml files in place; the IP repo paths don't change).
# Build: S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200

create_project -force scratch_rename_iodma_proj /tmp/scratch_rename_iodma_proj -part xczu7ev-ffvc1156-2-e

set BASE "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200"

# input IODMA (GenericPartition_0)
ipx::edit_ip_in_project -upgrade true -name tmp_edit_p0_idma -directory /tmp/ipedit_p0_idma \
    "$BASE/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip/component.xml"
set core [ipx::current_core]
set_property NAME p0_input_IODMA_hls_0 $core
set_property VLNV [regsub {:[^:]+:[^:]+$} [get_property VLNV $core] ":p0_input_IODMA_hls_0:1.0"] $core
set_property value {p0_input_IODMA_hls_0_v1_0} [ipx::get_user_parameters Component_Name -of_objects $core]
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project -delete

# output IODMA (GenericPartition_7)
ipx::edit_ip_in_project -upgrade true -name tmp_edit_p7_odma -directory /tmp/ipedit_p7_odma \
    "$BASE/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip/component.xml"
set core [ipx::current_core]
set_property NAME p7_output_IODMA_hls_0 $core
set_property VLNV [regsub {:[^:]+:[^:]+$} [get_property VLNV $core] ":p7_output_IODMA_hls_0:1.0"] $core
set_property value {p7_output_IODMA_hls_0_v1_0} [ipx::get_user_parameters Component_Name -of_objects $core]
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
close_project -delete

close_project
puts "Renamed both IODMA cores: p0_input_IODMA_hls_0, p7_output_IODMA_hls_0"
