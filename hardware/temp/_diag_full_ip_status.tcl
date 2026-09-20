open_project /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr
open_bd_design [get_files top.bd]
puts "=== report_ip_status ==="
report_ip_status -file /tmp/ip_status_detail.rpt
set fp [open /tmp/ip_status_detail.rpt r]
puts [read $fp]
close $fp
puts "=== end report ==="

puts "=== per-cell VLNV + IS_LOCKED ==="
foreach cell [get_bd_cells -hierarchical] {
    if {[get_property IS_LOCKED $cell] == 1} {
        puts "LOCKED: $cell  VLNV=[get_property VLNV $cell]  CONFIG.Component_Name=[get_property CONFIG.Component_Name $cell]"
    }
}
puts "=== end per-cell ==="
