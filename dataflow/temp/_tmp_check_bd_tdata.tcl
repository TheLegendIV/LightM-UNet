open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files top.bd]

set parts {GenericPartition_0_0 GenericPartition_1_0 GenericPartition_2_0 GenericPartition_3_0 GenericPartition_4_0 GenericPartition_6_0 GenericPartition_7_0}
foreach p $parts {
  puts "===== $p ====="
  set cell [get_bd_cells -quiet $p]
  if {$cell eq ""} { puts "  cell not found"; continue }
  foreach pin [get_bd_pins -quiet -of_objects $cell -filter {NAME =~ "*axis_0*"}] {
    set net [get_bd_nets -quiet -of_objects $pin]
    set width [get_property -quiet CONFIG.LEFT $pin]
    puts "  PIN $pin WIDTH(left)=$width NET=$net"
  }
  foreach intf [get_bd_intf_pins -quiet -of_objects $cell] {
    set net [get_bd_intf_nets -quiet -of_objects $intf]
    puts "  INTF $intf NET=$net"
  }
}
close_project
