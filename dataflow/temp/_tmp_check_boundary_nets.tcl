open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/impl_1/top_wrapper_routed.dcp

set parts {GenericPartition_0_0 GenericPartition_1_0 GenericPartition_2_0 GenericPartition_3_0 GenericPartition_4_0 GenericPartition_6_0 GenericPartition_7_0}

foreach p $parts {
  set cell [get_cells -quiet "top_i/$p"]
  if {$cell eq ""} {
    puts "PART $p: cell not found"
    continue
  }
  puts "===== $p ====="
  foreach pin [get_pins -quiet -of_objects $cell -filter {NAME =~ "*axis*" || NAME =~ "*ap_rst*" || NAME =~ "*ap_clk*"}] {
    set net [get_nets -quiet -of_objects $pin]
    if {$net eq ""} {
      puts "  PIN $pin : NO NET (floating)"
      continue
    }
    set driver [get_pins -quiet -of_objects $net -filter {DIRECTION == OUT}]
    set drivercell ""
    set driverref ""
    if {$driver ne ""} {
      set drivercell [get_cells -quiet -of_objects $driver]
      if {$drivercell ne ""} {
        set driverref [get_property -quiet REF_NAME $drivercell]
      }
    }
    set npins [llength [get_pins -quiet -of_objects $net]]
    puts "  PIN $pin : NET=$net DRIVER=$driver DRIVER_CELL=$drivercell DRIVER_REF=$driverref NET_PINS=$npins"
  }
}
close_project
