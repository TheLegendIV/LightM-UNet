open_project /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.xpr
puts "=== current ip_repo_paths ==="
puts [get_property ip_repo_paths [current_project]]
puts "=== bd cells matching GenericPartition ==="
puts [get_bd_cells -hierarchical -filter {VLNV =~ "*GenericPartition*"}]
puts "=== ip status (upgrade/generate state) for those cells ==="
foreach c [get_bd_cells -hierarchical -filter {VLNV =~ "*GenericPartition*"}] {
    set ip [get_ips -quiet -filter "IPDEF =~ {xilinx_finn:finn:GenericPartition*} && CELL_NAME == $c"]
    if {$ip ne ""} {
        puts "$c -> IP:$ip  IS_LOCKED=[get_property IS_LOCKED $ip]  UPGRADE_VERSIONS=[get_property UPGRADE_VERSIONS $ip] GENERATE_SYNTH_CHECKPOINT=[get_property GENERATE_SYNTH_CHECKPOINT $ip]"
    } else {
        puts "$c -> no matching IP object found via get_ips"
    }
}
puts "=== QUERY_DONE ==="
