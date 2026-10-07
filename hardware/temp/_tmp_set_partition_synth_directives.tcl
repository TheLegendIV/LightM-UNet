open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr

# Replicate FINN's own per-partition OOC synth directives on every auto-generated
# nested-IP synth run for the GenericPartition_* BD cells, instead of leaving
# them on Vivado's plain defaults.
set partition_runs [get_runs -filter {NAME =~ "*GenericPartition*_synth_1"}]
if {[llength $partition_runs] == 0} {
    puts "WARNING: no matching *GenericPartition*_synth_1 runs found"
}

foreach r $partition_runs {
    puts "Configuring $r"
    reset_run $r

    set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE                 PerformanceOptimized $r
    set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING                  true                 $r
    set_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION            one_hot              $r
    set_property STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS true                 $r
    set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING          off                  $r
    set_property STEPS.SYNTH_DESIGN.ARGS.NO_LC                     true                 $r
    set_property STEPS.SYNTH_DESIGN.ARGS.SHREG_MIN_SIZE            5                    $r

    puts "  DIRECTIVE                 = [get_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE $r]"
    puts "  RETIMING                  = [get_property STEPS.SYNTH_DESIGN.ARGS.RETIMING $r]"
    puts "  FSM_EXTRACTION            = [get_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION $r]"
    puts "  KEEP_EQUIVALENT_REGISTERS = [get_property STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS $r]"
    puts "  RESOURCE_SHARING          = [get_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING $r]"
    puts "  NO_LC                     = [get_property STEPS.SYNTH_DESIGN.ARGS.NO_LC $r]"
    puts "  SHREG_MIN_SIZE            = [get_property STEPS.SYNTH_DESIGN.ARGS.SHREG_MIN_SIZE $r]"
}

puts "Done. Re-launch synthesis for these runs (e.g. launch_runs \$partition_runs -jobs <N>; wait_on_runs \$partition_runs) then re-run top-level synth_1/impl_1."

close_project
