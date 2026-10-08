open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
set props {DIRECTIVE RETIMING FSM_EXTRACTION KEEP_EQUIVALENT_REGISTERS RESOURCE_SHARING NO_LC SHREG_MIN_SIZE FLATTEN_HIERARCHY MORE_OPTIONS}
foreach r [get_runs -filter {IS_SYNTHESIS == 1}] {
    set parts {}
    foreach p $props { lappend parts "$p=[get_property STEPS.SYNTH_DESIGN.ARGS.$p $r]" }
    puts "RUN $r | [join $parts { | }]"
}
close_project
