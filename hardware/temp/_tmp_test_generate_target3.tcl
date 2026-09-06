open_project /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.xpr
open_bd_design /tmp/finn_dev_thelegendiv/combined_stitch_proj_k3gshz1u/finn_design.srcs/sources_1/bd/finn_design/finn_design.bd
puts "=== trying upgrade_ip ==="
if {[catch {upgrade_ip [get_bd_cells]} err]} {
    puts "ERR_UPGRADE: $err"
} else {
    puts "OK_UPGRADE"
}
puts "=== trying generate_target after upgrade ==="
if {[catch {generate_target -force {synthesis} [get_files finn_design.bd]} err]} {
    puts "ERR_GEN: $err"
} else {
    puts "OK_GEN"
}
puts "=== validate again ==="
validate_bd_design
