open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/impl_1/top_wrapper_routed.dcp
set fp [open /tmp/partition_pin_report.txt w]
foreach n {0 1 2 3 4 5 6 7} {
    set cellpath "top_i/GenericPartition_${n}_0/inst"
    if {[llength [get_cells -quiet $cellpath]] == 0} {
        puts $fp "=== $cellpath NOT FOUND ==="
        continue
    }
    puts $fp "=== $cellpath ==="
    foreach p [get_pins -of_objects [get_cells $cellpath] -filter {DIRECTION == IN}] {
        set net [get_nets -quiet -of_objects $p]
        set drivername "NONE"
        set is_const "no"
        if {[llength $net] > 0} {
            set driver_pins [get_pins -quiet -of_objects $net -filter {DIRECTION == OUT}]
            if {[llength $driver_pins] > 0} {
                set drivername $driver_pins
            }
            if {[string match "*GLOBAL_LOGIC*" $net] || [string match "*<const*" $net]} {
                set is_const "yes"
            }
        }
        puts $fp "  PIN $p  NET=$net  DRIVER=$drivername  CONST=$is_const"
    }
}
close $fp
exit
