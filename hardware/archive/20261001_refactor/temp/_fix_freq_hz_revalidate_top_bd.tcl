open_project {/home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr}
update_ip_catalog -rebuild
open_bd_design [get_files top.bd]
validate_bd_design -force
save_bd_design
generate_target all [get_files top.bd] -force
close_project
