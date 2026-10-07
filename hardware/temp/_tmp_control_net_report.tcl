open_checkpoint /home/thelegendiv/finn/vivado_projects/S12_256_analytical/s12_256_analytical/s12_256_analytical.runs/impl_1/top_wrapper_routed.dcp
set fp [open /tmp/control_net_report.txt w]
foreach sig {ap_clk ap_rst_n s_axis_0_tvalid m_axis_0_tvalid m_axis_0_tready} {
    set netpath "top_i/GenericPartition_0_0/${sig}"
    set net [get_nets -quiet $netpath]
    puts $fp "=== NET $netpath ==="
    if {[llength $net] == 0} {
        puts $fp "  NET OBJECT NOT FOUND"
        continue
    }
    puts $fp "  TYPE=[get_property TYPE $net] "
    foreach p [get_pins -quiet -of_objects $net] {
        puts $fp "  PIN $p DIR=[get_property DIRECTION $p]"
    }
    foreach p [get_ports -quiet -of_objects $net] {
        puts $fp "  PORT $p"
    }
}
puts $fp "=== m_axis_0_tdata search (partition0 output) ==="
foreach p [get_pins -quiet -of_objects [get_cells top_i/GenericPartition_0_0/inst] -filter {DIRECTION==OUT}] {
    set net [get_nets -quiet -of_objects $p]
    puts $fp "  PIN $p NET=$net"
}
close $fp
exit
