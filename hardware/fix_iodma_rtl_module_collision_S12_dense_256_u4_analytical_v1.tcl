# STEP 1 of 2: fixes the RTL-level IODMA_hls_0 Verilog MODULE NAME
# collision for S12_dense_256_u4_analytical_v1 (both p0_input and
# p7_output IODMA cores compile, via Vitis HLS, to the identical literal
# module name "IODMA_hls_0" with incompatible port lists -- out_V_* vs
# in0_V_* -- even though fix_iodma_name_conflict_S12_dense_256_u4_analytical_v1.tcl
# already disambiguated their IP-XACT NAME/VLNV. Vivado's combined
# synth_1 elaborates all partitions' raw Verilog in one global namespace,
# so the two still collide at the literal "module IODMA_hls_0" level --
# confirmed via "ERROR: [Synth 8-448] named port connection 'in0_V_TVALID'
# does not exist for instance 'inst' of module 'IODMA_hls_0'" in
# top_p7_output_IODMA_hls_0_0_0.v).
#
# This overwrites each packaged IP's hdl/verilog sources (the copies
# actually referenced by its component.xml fileset, NOT the pristine
# sol1/impl/verilog HLS output) with the already-renamed sources from
# hardware/_rename_iodma_sources.py's prior output (module declaration +
# every internal submodule reference consistently prefixed
# p0_input_/p7_output_), then refreshes IP-XACT checksums.
#
# Run ONCE, BEFORE hardware/regen_iodma_bd_outputs_S12_dense_256_u4_analytical_v1.tcl.
# Does not touch the live S12_256_analytical project; only the IP repo.

set BASE "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200"
set RENAMED "/home/thelegendiv/finn/notebooks/enet/vivado_projects/zcu7ev_S12_dense_256_u4_analytical_v1_srcs/iodma_renamed"

proc swap_ip_sources {tag component_xml renamed_dir} {
    set proj_name "tmp_swap_$tag"
    set proj_dir "/tmp/$proj_name"
    set edit_dir "/tmp/${proj_name}_edit"
    create_project -force $proj_name $proj_dir -part xczu7ev-ffvc1156-2-e
    ipx::edit_ip_in_project -upgrade true -name tmp_edit -directory $edit_dir $component_xml
    set core [ipx::current_core]
    set ip_dir [file dirname $component_xml]
    set vdir "$ip_dir/hdl/verilog"
    set n 0
    foreach f [glob -nocomplain "$renamed_dir/*.v"] {
        file copy -force $f "$vdir/[file tail $f]"
        incr n
    }
    puts "\[$tag\] copied $n renamed Verilog files into $vdir"
    ipx::update_checksums $core
    ipx::save_core $core
    close_project -delete
    close_project
}

swap_ip_sources p0_input \
    "$BASE/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip/component.xml" \
    "$RENAMED/p0_input"

swap_ip_sources p7_output \
    "$BASE/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip/component.xml" \
    "$RENAMED/p7_output"

puts "FIX_RTL_COLLISION_DONE: both IODMA cores repackaged with collision-free module names."
