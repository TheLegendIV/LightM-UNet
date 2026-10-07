open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files top.bd]
set c [get_bd_cells GenericPartition_1_0]
puts "VLNV=[get_property VLNV $c]"
foreach prop [list_property $c] {
  if {[string match "CONFIG.*" $prop]} {
    puts "  $prop = [get_property $prop $c]"
  }
}
close_project
