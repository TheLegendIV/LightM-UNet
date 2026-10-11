open_project -read_only /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
open_bd_design [get_files */top.bd]
puts "=== CELLS ==="
foreach c [get_bd_cells] { puts "CELL [get_property NAME $c] [get_property VLNV $c]" }
puts "=== FINN/IODMA intf pins ==="
foreach c [get_bd_cells -filter {VLNV =~ "*GenericPartition*" || VLNV =~ "*IODMA*"}] {
    foreach p [get_bd_intf_pins -of_objects $c] {
        set net [get_bd_intf_nets -quiet -of_objects $p]
        puts "PIN [get_property PATH $p] mode=[get_property MODE $p] net=$net"
    }
}
puts "=== ip_repo_paths not under finn_build_tmp ==="
foreach p [get_property ip_repo_paths [current_project]] { if {[string first finn_build_tmp $p] < 0} { puts "REPO $p" } }
puts "=== board/part ==="
puts "[get_property part [current_project]] [get_property board_part [current_project]]"
puts "=== runs ==="
foreach r [get_runs] { puts "$r [get_property STATUS $r]" }
puts "=== files ==="
foreach f [get_files -quiet] { puts "FILE $f" }
close_project
