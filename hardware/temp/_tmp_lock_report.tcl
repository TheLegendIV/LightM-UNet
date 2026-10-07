open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/impl_1/top_wrapper_routed.dcp
set fp [open /tmp/lock_report.txt w]
foreach n {0 1 7} {
  set c [get_cells top_i/GenericPartition_${n}_0]
  puts $fp "=== $c ==="
  puts $fp "  IS_LOCKED=[get_property -quiet IS_LOCKED $c]"
  puts $fp "  IS_BLACKBOX=[get_property -quiet IS_BLACKBOX $c]"
  puts $fp "  DONT_TOUCH=[get_property -quiet DONT_TOUCH $c]"
  puts $fp "  KEEP_HIERARCHY=[get_property -quiet KEEP_HIERARCHY $c]"
  set inst [get_cells -quiet ${c}/inst]
  if {[llength $inst]} {
    puts $fp "  inst IS_LOCKED=[get_property -quiet IS_LOCKED $inst]"
    puts $fp "  inst DONT_TOUCH=[get_property -quiet DONT_TOUCH $inst]"
  }
}
close $fp
exit
