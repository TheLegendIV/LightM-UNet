open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
update_ip_catalog -rebuild
open_bd_design [get_files top.bd]

set cells [get_bd_cells -filter {NAME =~ "StreamingDataflowPar*"}]
puts "found [llength $cells] partition-like cells"
foreach c $cells {
    if {[llength [get_bd_pins -quiet $c/ap_clk]] > 0} {
        set p [get_bd_pins $c/ap_clk]
        set net [get_bd_nets -of_objects $p]
        set src [get_bd_pins -of_objects $net -filter {DIR == "O"}]
        puts "$p driven by net $net, source pin(s): $src"
        disconnect_bd_net $net $p
        connect_bd_net $src $p
        puts "reconnected $p to $src"
    } else {
        puts "no ap_clk pin on $c, skipping"
    }
}

validate_bd_design -force
save_bd_design
generate_target all [get_files top.bd] -force
close_project
