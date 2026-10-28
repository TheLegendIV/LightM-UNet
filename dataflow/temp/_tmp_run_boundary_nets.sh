source /tools/Xilinx/Vivado/2022.2/settings64.sh
vivado -mode batch -source /tmp/check_boundary_nets.tcl -log /tmp/check_boundary_nets.log -journal /tmp/check_boundary_nets.jou
