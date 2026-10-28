# STEP 1b (follow-up to fix_iodma_rtl_module_collision_...): updates the
# IP-XACT <spirit:modelName> field in both IODMA cores' component.xml.
# This field is SEPARATE from NAME/VLNV/Component_Name (already fixed by
# fix_iodma_name_conflict_S12_dense_256_u4_analytical_v1.tcl) and from the
# actual Verilog module declaration (already fixed by
# fix_iodma_rtl_module_collision_S12_dense_256_u4_analytical_v1.tcl) --
# it is what Vivado's BD wrapper generator actually reads to decide which
# module name to instantiate in the generated top_p*_..._0_0.v wrapper.
# Left stale at "IODMA_hls_0" it caused "module 'IODMA_hls_0' not found"
# after the Verilog source rename (previously "named port connection...
# does not exist" before the source rename, since it was picking up the
# WRONG same-named module from the other partition).

set BASE "/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/S12_dense_256_u4_analytical_v1_ft15ep_int6_fps250_lat200"

proc fix_model_name {tag component_xml new_model_name} {
    set proj_name "tmp_modelname_$tag"
    set proj_dir "/tmp/$proj_name"
    set edit_dir "/tmp/${proj_name}_edit"
    create_project -force $proj_name $proj_dir -part xczu7ev-ffvc1156-2-e
    ipx::edit_ip_in_project -upgrade true -name tmp_edit -directory $edit_dir $component_xml
    set core [ipx::current_core]
    set n 0
    foreach view [ipx::get_views -of_objects $core] {
        if {[get_property MODEL_NAME $view] ne ""} {
            set_property MODEL_NAME $new_model_name $view
            incr n
        }
    }
    puts "\[$tag\] updated MODEL_NAME on $n view(s) to $new_model_name"
    ipx::update_checksums $core
    ipx::save_core $core
    close_project -delete
    close_project
}

fix_model_name p0_input \
    "$BASE/GenericPartition_0/code_gen_ipgen_IODMA_hls_0_2aozvq23/project_IODMA_hls_0/sol1/impl/ip/component.xml" \
    p0_input_IODMA_hls_0

fix_model_name p7_output \
    "$BASE/GenericPartition_7/code_gen_ipgen_IODMA_hls_0_094h060g/project_IODMA_hls_0/sol1/impl/ip/component.xml" \
    p7_output_IODMA_hls_0

puts "FIX_MODEL_NAME_DONE"
