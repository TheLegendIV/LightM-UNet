open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
open_bd_design [get_files top.bd]

set cells [get_bd_cells -filter {NAME =~ "StreamingDataflowPar*"}]
puts "found [llength $cells] partition cells"
foreach c $cells {
    set p [get_bd_pins $c/ap_clk]
    set_property CONFIG.FREQ_TOLERANCE_HZ 20000 $p
    puts "set tolerance on $p"
}

validate_bd_design -force
save_bd_design
generate_target all [get_files top.bd] -force
close_project
