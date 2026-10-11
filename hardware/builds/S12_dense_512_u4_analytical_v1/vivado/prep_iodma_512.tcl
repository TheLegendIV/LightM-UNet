# Repackages the two 512 IODMA IP cores in place (same fixes as the 256
# fix_iodma_name_conflict / fix_iodma_rtl_module_collision / fix_iodma_model_name
# scripts, merged into one pass per core):
#   1. IP NAME / VLNV / Component_Name -> p0_input_IODMA_hls_0 / p7_output_IODMA_hls_0
#   2. hdl/verilog sources overwritten with the prefixed ones from
#      rename_iodma_sources_512.py
#   3. MODEL_NAME on all views -> new module name
# Usage: vivado -mode batch -source prep_iodma_512.tcl -tclargs <finn_build_dir> <renamed_dir>

set BASE [lindex $argv 0]
set RENAMED [lindex $argv 1]

proc repackage {tag part prefix} {
    global BASE RENAMED
    set hits [glob -nocomplain "$BASE/$part/code_gen_ipgen_IODMA_hls_0_*/project_IODMA_hls_0/sol1/impl/ip/component.xml"]
    if {[llength $hits] != 1} { error "expected 1 IODMA component.xml for $part, got: $hits" }
    set component_xml [lindex $hits 0]
    set new_name ${prefix}_IODMA_hls_0
    set proj tmp_prep_$tag
    create_project -force $proj /tmp/$proj -part xczu7ev-ffvc1156-2-e
    ipx::edit_ip_in_project -upgrade true -name tmp_edit -directory /tmp/${proj}_edit $component_xml
    set core [ipx::current_core]

    set_property NAME $new_name $core
    set_property VLNV [regsub {:[^:]+:[^:]+$} [get_property VLNV $core] ":${new_name}:1.0"] $core
    set_property value ${new_name}_v1_0 [ipx::get_user_parameters Component_Name -of_objects $core]

    set vdir "[file dirname $component_xml]/hdl/verilog"
    set n 0
    foreach f [glob -nocomplain "$RENAMED/$tag/*.v"] {
        file copy -force $f "$vdir/[file tail $f]"
        incr n
    }
    puts "\[$tag\] copied $n renamed Verilog files into $vdir"

    set m 0
    foreach fg [ipx::get_file_groups -of_objects $core] {
        if {[get_property MODEL_NAME $fg] ne ""} {
            set_property MODEL_NAME $new_name $fg
            incr m
        }
    }
    puts "\[$tag\] MODEL_NAME set on $m file group(s) -> $new_name"

    ipx::create_xgui_files $core
    ipx::update_checksums $core
    ipx::save_core $core
    puts "\[$tag\] VLNV now [get_property VLNV $core]"
    close_project -delete
    close_project
}

repackage p0_input GenericPartition_0 p0_input
repackage p7_output GenericPartition_7 p7_output
puts "PREP_IODMA_512_DONE"
