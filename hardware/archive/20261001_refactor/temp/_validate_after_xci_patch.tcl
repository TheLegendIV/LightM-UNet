open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
open_bd_design [get_files top.bd]

foreach i {0 1 2 3 4 5 6 7} {
    set p [get_bd_pins /StreamingDataflowPar_$i/ap_clk]
    puts "P$i FREQ_HZ = [get_property CONFIG.FREQ_HZ $p]"
}

validate_bd_design -force
save_bd_design
close_project
