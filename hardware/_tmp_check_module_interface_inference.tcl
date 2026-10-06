# Empirical check: does Vivado auto-infer a proper AXI4-Lite bus
# interface on a flat "module" BD cell referencing the raw IODMA
# Verilog (no IP-XACT packaging), or does it only expose flat wires?
# This determines whether assign_bd_address can even generate address
# decode for it -- independent of any Vitis driver/xparameters.h concern.
# Read-only/throwaway: creates a scratch BD, checks, discards (no save).

open_project /home/thelegendiv/finn/notebooks/enet/vivado_projects/zcu7ev_S12_dense_256_u4_analytical_v1/zcu7ev_S12_dense_256_u4_analytical_v1.xpr

create_bd_design "scratch_iface_check"
create_bd_cell -type module -reference p0_input_IODMA_hls_0 iodma_test_0

puts "---- bd_intf_pins (bundled interfaces Vivado recognized) ----"
puts [get_bd_intf_pins -of_objects [get_bd_cells iodma_test_0] -quiet]

puts "---- bd_pins count (flat individual wires) ----"
puts [llength [get_bd_pins -of_objects [get_bd_cells iodma_test_0] -quiet]]

puts "SCRATCH_CHECK_DONE"
close_project
