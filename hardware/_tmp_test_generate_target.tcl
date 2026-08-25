<<<<<<< HEAD
open_project /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.xpr
open_bd_design /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.srcs/sources_1/bd/finn_design/finn_design.bd
set bd_cells [get_bd_cells]
puts "BD CELLS: $bd_cells"
foreach c $bd_cells {
    puts "--- $c ---"
    puts [get_property IP_FILE $c]
    puts [get_property CONFIG.Component_Name $c]
}
puts "=== trying generate_target synthesis on one cell ==="
if {[catch {generate_target {synthesis} [get_files finn_design.bd] -force} err]} {
    puts "ERR1: $err"
} else {
    puts "OK1"
}
=======
open_project /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.xpr
open_bd_design /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.srcs/sources_1/bd/finn_design/finn_design.bd
set bd_cells [get_bd_cells]
puts "BD CELLS: $bd_cells"
foreach c $bd_cells {
    puts "--- $c ---"
    puts [get_property IP_FILE $c]
    puts [get_property CONFIG.Component_Name $c]
}
puts "=== trying generate_target synthesis on one cell ==="
if {[catch {generate_target {synthesis} [get_files finn_design.bd] -force} err]} {
    puts "ERR1: $err"
} else {
    puts "OK1"
}
>>>>>>> 1c37749cf21da213659e029bae27ca2f6f8981fe
