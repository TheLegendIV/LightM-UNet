set proj_path /home/thelegendiv/finn/notebooks/enet/finn_build_tmp/vivado_zynq_proj_ivzy_6_z/finn_zynq_link.xpr
open_project $proj_path
open_bd_design [get_files top.bd]

puts "=== smartconnect_0 pins ==="
foreach pin [get_bd_intf_pins -of_objects [get_bd_cells /smartconnect_0]] {
    puts "  $pin"
}

puts "=== what's connected to /smartconnect_0/S01_AXI ==="
set net [get_bd_intf_nets -of_objects [get_bd_intf_pins /smartconnect_0/S01_AXI]]
puts "net: $net"
set pins [get_bd_intf_pins -of_objects $net]
puts "pins on net: $pins"

puts "=== FREQ_HZ on all clk pins of interest ==="
foreach cellname {StreamingDataflowPar_8 StreamingDataflowPar_9 smartconnect_0 zynq_ultra_ps_e_0} {
    set clkpins [get_bd_pins -of_objects [get_bd_cells /$cellname] -filter {TYPE==clk}]
    foreach cp $clkpins {
        if {[catch {set f [get_property CONFIG.FREQ_HZ $cp]} err]} {
            puts "  $cp  FREQ_HZ=<none/err: $err>"
        } else {
            puts "  $cp  FREQ_HZ=$f"
        }
    }
}
