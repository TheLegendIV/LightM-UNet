open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
open_bd_design [get_files top.bd]

set psclk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
puts "--- properties on $psclk before ---"
foreach prop [list_property $psclk] {
    if {[string match "CONFIG.FREQ*" $prop]} {
        puts "$prop = [get_property $prop $psclk]"
    }
}

catch {set_property CONFIG.FREQ_TOLERANCE_HZ {30000} $psclk} msg
puts "set attempt result: $msg"

puts "--- properties on $psclk after ---"
foreach prop [list_property $psclk] {
    if {[string match "CONFIG.FREQ*" $prop]} {
        puts "$prop = [get_property $prop $psclk]"
    }
}

validate_bd_design -force
save_bd_design
generate_target all [get_files top.bd] -force
close_project
