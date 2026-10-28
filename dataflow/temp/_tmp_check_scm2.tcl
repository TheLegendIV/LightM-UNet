open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
foreach n {0 1 2 3 4 6 7} {
  set path "*/sources_1/bd/top/ip/top_GenericPartition_${n}_0_0/top_GenericPartition_${n}_0_0.xci"
  set xci [get_files -quiet $path]
  if {$xci eq ""} {
    puts "PART=$n  xci not found"
    continue
  }
  puts "PART=$n SYNTH_CHECKPOINT_MODE=[get_property -quiet SYNTH_CHECKPOINT_MODE $xci] IS_LOCKED=[get_property -quiet IS_LOCKED $xci] IS_GLOBAL_INCLUDE=[get_property -quiet IS_GLOBAL_INCLUDE $xci] GEN_SYNTH=[get_property -quiet GENERATE_SYNTH_CHECKPOINT $xci]"
}
close_project
