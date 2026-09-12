# Usage with Vitis IDE:
# In Vitis IDE create a Single Application Debug launch configuration,
# change the debug type to 'Attach to running target' and provide this 
# tcl script in 'Execute Script' option.
# Path of this script: C:\DEV\repos\LightM-UNet\deployment\FINN_interface\lwip_zcu106_preset_system\_ide\scripts\debugger_lwip_zcu106_preset-default.tcl
# 
# 
# Usage with xsct:
# To debug using xsct, launch xsct and run below command
# source C:\DEV\repos\LightM-UNet\deployment\FINN_interface\lwip_zcu106_preset_system\_ide\scripts\debugger_lwip_zcu106_preset-default.tcl
# 
connect -url tcp:127.0.0.1:3121
source F:/PLD/Xilinx/Vitis/2022.2/scripts/vitis/util/zynqmp_utils.tcl
targets -set -nocase -filter {name =~"APU*"}
rst -system
after 3000
targets -set -filter {jtag_cable_name =~ "Xilinx HW-FTDI-TEST FT232H 84713" && level==0 && jtag_device_ctx=="jsn-HW-FTDI-TEST FT232H-84713-14730093-0"}
fpga -file C:/DEV/repos/LightM-UNet/deployment/FINN_interface/lwip_zcu106_preset/_ide/bitstream/zcu106_preset_applied.bit
targets -set -nocase -filter {name =~"APU*"}
loadhw -hw C:/DEV/repos/LightM-UNet/deployment/FINN_interface/zcu106_test/export/zcu106_test/hw/zcu106_preset_applied.xsa -mem-ranges [list {0x80000000 0xbfffffff} {0x400000000 0x5ffffffff} {0x1000000000 0x7fffffffff}] -regs
configparams force-mem-access 1
targets -set -nocase -filter {name =~"APU*"}
set mode [expr [mrd -value 0xFF5E0200] & 0xf]
targets -set -nocase -filter {name =~ "*A53*#0"}
rst -processor
dow C:/DEV/repos/LightM-UNet/deployment/FINN_interface/zcu106_test/export/zcu106_test/sw/zcu106_test/boot/fsbl.elf
set bp_8_0_fsbl_bp [bpadd -addr &XFsbl_Exit]
con -block -timeout 60
bpremove $bp_8_0_fsbl_bp
targets -set -nocase -filter {name =~ "*A53*#0"}
rst -processor
dow C:/DEV/repos/LightM-UNet/deployment/FINN_interface/lwip_zcu106_preset/Debug/lwip_zcu106_preset.elf
configparams force-mem-access 0
bpadd -addr &main
