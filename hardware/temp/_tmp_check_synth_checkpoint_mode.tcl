open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files top.bd]
foreach xci [get_files -quiet "*/top_GenericPartition_*.xci"] {
  puts "XCI=$xci SYNTH_CHECKPOINT_MODE=[get_property -quiet SYNTH_CHECKPOINT_MODE $xci] IS_LOCKED=[get_property -quiet IS_LOCKED $xci] GEN_SYNTH=[get_property -quiet GENERATE_SYNTH_CHECKPOINT $xci]"
}
foreach c [get_bd_cells -filter {VLNV =~ "xilinx_finn:finn:GenericPartition_*"}] {
  puts "CELL=$c IS_LOCKED(cell)=[get_property -quiet IS_LOCKED $c]"
}
close_project
