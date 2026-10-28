# Step 1 of the hand-built ZCU106 combine: project + Zynq PS + AXI
# interconnect/smartconnect BD only (no partition/IODMA sources yet --
# those are added in step 3 as flat Verilog, never as BD cells, to avoid
# the documented "Cannot upgrade to invalid target ''" Vivado bug hit
# when combining FINN's partition-level stitched IPs via IP Integrator
# (see hardware/finn_partition_build_steps.py step_combine_partitions).
# The PS/interconnect/smartconnect here are plain catalog IP, so BD is
# safe for them.

set PROJ_NAME zcu7ev_S12_dense_256_u4_analytical_v1
set PROJ_DIR /home/thelegendiv/finn/notebooks/enet/vivado_projects/$PROJ_NAME
set FPGA_PART xczu7ev-ffvc1156-2-e
set NUM_AXILITE 2
set NUM_AXIMM 2
set FREQ_MHZ 100

file mkdir /home/thelegendiv/finn/notebooks/enet/vivado_projects
create_project $PROJ_NAME $PROJ_DIR -part $FPGA_PART -force

set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
lappend paths_prop /home/thelegendiv/finn/deps/xil-bdf/boards
lappend paths_prop /home/thelegendiv/finn/deps/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_prop

set_property board_part xilinx.com:zcu106:part0:2.6 [current_project]

create_bd_design "ps_interconnect"

set zynq_ps_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:zynq_ultra_ps_e:*"]]
create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells zynq_ps]
set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {1}] [get_bd_cells zynq_ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ps]
set_property -dict [list CONFIG.PSU__OVERRIDE__BASIC_CLOCK {0}] [get_bd_cells zynq_ps]
set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]

set interconnect_vlnv [get_property VLNV [get_ipdefs -all "xilinx.com:ip:axi_interconnect:*" -filter design_tool_contexts=~*IPI*]]
set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_0
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM] [get_bd_cells smartconnect_0]
set_property -dict [list CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]

connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}} [get_bd_pins axi_interconnect_0/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}} [get_bd_pins axi_interconnect_0/S00_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}} [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins smartconnect_0/aresetn]
# smartconnect_0/aclk already tied to zynq_ps/pl_clk0 by the ACLK automation above

# expose the 2 gmem master-in ports (smartconnect slave side) and the 2
# axi-lite control master-out ports (interconnect master side) at the BD
# boundary -- these get wired, at the flat top-level Verilog in step 3,
# directly to the two standalone IODMA cores (never as BD cells).
make_bd_intf_pins_external [get_bd_intf_pins smartconnect_0/S00_AXI]
make_bd_intf_pins_external [get_bd_intf_pins smartconnect_0/S01_AXI]
make_bd_intf_pins_external [get_bd_intf_pins axi_interconnect_0/M00_AXI]
make_bd_intf_pins_external [get_bd_intf_pins axi_interconnect_0/M01_AXI]

# M*_ACLK automation needs the master ports actually connected to
# something (now satisfied via the external ports above) or it errors
# out with "Automation rule ... was not applied" on dangling masters.
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} } [get_bd_pins axi_interconnect_0/M*_ACLK]

save_bd_design
assign_bd_address
validate_bd_design

set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [get_files ps_interconnect.bd]
make_wrapper -files [get_files ps_interconnect.bd] -import -fileset sources_1 -top

save_bd_design
puts "STEP1_DONE: project at $PROJ_DIR"
close_project
