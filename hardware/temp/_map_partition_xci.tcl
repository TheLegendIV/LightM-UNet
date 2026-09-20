open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
open_bd_design [get_files top.bd]

for {set i 0} {$i <= 7} {incr i} {
    set c [get_bd_cells /StreamingDataflowPar_$i]
    set comp [get_property CONFIG.Component_Name $c]
    puts "MAP $i -> $comp"
}
close_project
