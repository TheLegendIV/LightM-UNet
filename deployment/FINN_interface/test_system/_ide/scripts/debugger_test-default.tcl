# Usage with Vitis IDE:
# In Vitis IDE create a Single Application Debug launch configuration,
# change the debug type to 'Attach to running target' and provide this 
# tcl script in 'Execute Script' option.
# Path of this script: C:\DEV\repos\LightM-UNet\deployment\FINN_interface\test_system\_ide\scripts\debugger_test-default.tcl
# 
# 
# Usage with xsct:
# To debug using xsct, launch xsct and run below command
# source C:\DEV\repos\LightM-UNet\deployment\FINN_interface\test_system\_ide\scripts\debugger_test-default.tcl
# 
connect -url tcp:127.0.0.1:3121
source F:/PLD/Xilinx/Vitis/2022.2/scripts/vitis/util/zynqmp_utils.tcl
targets -set -nocase -filter {name =~"APU*"}
rst -system
after 3000
targets -set -nocase -filter {name =~"APU*"}
loadhw -hw C:/DEV/repos/LightM-UNet/deployment/FINN_interface/zcu106/export/zcu106/hw/zcu106.xsa -mem-ranges [list {0x80000000 0xbfffffff} {0x400000000 0x5ffffffff} {0x1000000000 0x7fffffffff}] -regs
configparams force-mem-access 1
targets -set -nocase -filter {name =~"APU*"}
set mode [expr [mrd -value 0xFF5E0200] & 0xf]
targets -set -nocase -filter {name =~ "*A53*#0"}
rst -processor
dow C:/DEV/repos/LightM-UNet/deployment/FINN_interface/zcu106/export/zcu106/sw/zcu106/boot/fsbl.elf
set bp_24_47_fsbl_bp [bpadd -addr &XFsbl_Exit]
con -block -timeout 60
bpremove $bp_24_47_fsbl_bp
targets -set -nocase -filter {name =~ "*A53*#0"}
rst -processor
dow C:/DEV/repos/LightM-UNet/deployment/FINN_interface/test/Debug/test.elf
configparams force-mem-access 0
bpadd -addr &main
