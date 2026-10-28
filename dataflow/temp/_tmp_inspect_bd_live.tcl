open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files top.bd]
puts "=== CELLS ==="
foreach c [get_bd_cells] {
    puts [format "CELL %s  VLNV=%s" $c [get_property VLNV $c]]
}
puts "=== IP_REPO_PATHS ==="
puts [get_property ip_repo_paths [current_project]]
puts "=== INTF NETS ==="
foreach n [get_bd_intf_nets] {
    puts [format "INTFNET %s :: %s" $n [get_bd_intf_pins -of_objects $n]]
}
puts "=== NETS (non-intf) ==="
foreach n [get_bd_nets] {
    puts [format "NET %s :: %s" $n [get_bd_pins -of_objects $n]]
}
puts "=== PORTS of each GenericPartition/IODMA cell ==="
foreach c [get_bd_cells] {
    if {[string match "GenericPartition_*" $c] || [string match "*IODMA*" $c]} {
        puts "--- $c ---"
        foreach p [get_bd_intf_pins -of_objects $c] { puts "  INTFPIN $p" }
        foreach p [get_bd_pins -of_objects $c] { puts "  PIN $p" }
    }
}
close_project
