open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/synth_1/top_wrapper.dcp
set fp [open /tmp/synth1_lock_report.txt w]
foreach n {0 1 7} {
  set c [get_cells -quiet top_i/GenericPartition_${n}_0]
  puts $fp "=== $c ==="
  if {[llength $c] == 0} { puts $fp "  CELL NOT FOUND"; continue }
  puts $fp "  IS_LOCKED=[get_property -quiet IS_LOCKED $c]"
}
puts $fp "=== m_axis_0_tdata search partition0 (post synth_1, pre opt_design) ==="
foreach p [get_pins -quiet -of_objects [get_cells -quiet top_i/GenericPartition_0_0/inst] -filter {DIRECTION==OUT}] {
    set net [get_nets -quiet -of_objects $p]
    puts $fp "  PIN $p NET=$net"
}
close $fp
exit
