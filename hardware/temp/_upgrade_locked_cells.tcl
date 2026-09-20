set proj_path /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr
open_project $proj_path
open_bd_design [get_files top.bd]

set all_cells [get_bd_cells /StreamingDataflowPar_*]
puts "Cells to check: $all_cells"
foreach c $all_cells {
    puts "  $c  IS_LOCKED=[get_property IS_LOCKED $c]  VLNV=[get_property VLNV $c]"
}

puts "=== upgrading locked cells ==="
if {[catch {upgrade_bd_cells $all_cells} err]} {
    puts "upgrade_bd_cells ERROR: $err"
} else {
    puts "upgrade_bd_cells OK"
}

puts "=== validate_bd_design ==="
if {[catch {validate_bd_design -force} err2]} {
    puts "validate ERROR: $err2"
} else {
    puts "validate OK"
}

puts "=== save_bd_design ==="
if {[catch {save_bd_design} err3]} {
    puts "save ERROR: $err3"
} else {
    puts "save OK"
}

puts "=== final report_ip_status ==="
report_ip_status -file /tmp/ip_status_final.rpt
set fp [open /tmp/ip_status_final.rpt r]
foreach line [split [read $fp] "\n"] {
    if {[string match "*not found*" $line] || [string match "*locked*" $line] || [string match "*Instance Name*" $line]} {
        puts $line
    }
}
close $fp
